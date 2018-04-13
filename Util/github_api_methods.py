from datetime import datetime

from github import Github

from gitMine.VCClasses import Comment, Commit, IssueStates, PullRequest, Issue, StateChange


def parse_comment_ref(comment_ref):
    return Comment(id_=comment_ref.id, author=comment_ref.user.name if comment_ref.user else None,
                   body=comment_ref.body, timestamp=comment_ref.created_at.replace(tzinfo=None))


def parse_commit_ref(commit_ref, project):
    msg = commit_ref.commit.message
    title = msg[:min(80, len(msg))]
    desc = msg[min(80, len(msg)):]
    diffs = list()
    for file_ref in commit_ref.files:
        diffs.append(parse_file_ref(file_ref))
    return Commit(parent=[p.sha for p in commit_ref.parents],
                  author=commit_ref.author.name if commit_ref.author else None,
                  branches=None, c_hash=commit_ref.sha, desc=desc, diff=diffs,
                  repository=project.replace('_', '/'), timestamp=datetime.strptime(commit_ref.last_modified,
                                                                                    '%a, %d %b %Y %H:%M:%S GMT')
                  .replace(tzinfo=None),
                  title=title)


def parse_file_ref(file_ref):
    return '%s' % file_ref.filename, '%s' % file_ref.patch


def parse_label_ref(label_ref):
    return label_ref.name if label_ref else None


def parse_pr_ref(pr_ref, project):
    author = pr_ref.user.name if pr_ref.user else None

    comments = [Comment(id_='issuecomment_%d' % pr_ref.number, author=author, body=pr_ref.body,
                        timestamp=pr_ref.created_at.replace(tzinfo=None))]
    for comment_ref in pr_ref.get_comments():
        comments.append(parse_comment_ref(comment_ref))

    commits = list()
    for commit_ref in pr_ref.get_commits():
        commits.append(parse_commit_ref(commit_ref, project))

    diffs = list()
    for file_ref in pr_ref.get_files():
        diffs.append(parse_file_ref(file_ref))

    state = pr_ref.merged
    closed_at = pr_ref.closed_at
    if state:
        state = IssueStates.merged
    elif closed_at:
        state = IssueStates.closed
    else:
        state = IssueStates.open

    assignee = pr_ref.assignee
    assignee = assignee.name if assignee is not None else None

    return PullRequest(number='issue_%d' % pr_ref.number, assignee=assignee,
                       comments=comments, commits=commits, diffs=diffs, from_branch=pr_ref.head.ref,
                       from_repo=pr_ref.head.repo.name if pr_ref.head.repo else None,
                       to_branch=pr_ref.base.ref,
                       to_repo=pr_ref.base.repo.name if pr_ref.base.repo else None,
                       labels=list(), state=state, title=pr_ref.title if pr_ref.title is not None else '')


def parse_event(event_ref):
    timestamp = event_ref.created_at.replace(tzinfo=None)
    by = event_ref.actor.login
    if event_ref.event == 'referenced':
        return event_ref.commit_id
    elif event_ref.event == 'closed':
        return StateChange(from_=IssueStates.open, to_=IssueStates.closed, by=by,
                           timestamp=timestamp, commit=event_ref.commit_id)
    elif event_ref.event == 'reopened':
        return StateChange(from_=IssueStates.closed, to_=IssueStates.open, by=by,
                           timestamp=timestamp, commit=event_ref.commit_id)
    elif (event_ref.event == 'merged') or (event_ref.event == 'marked_as_duplicate'):
        return StateChange(from_=IssueStates.open, to_=IssueStates.merged, by=by,
                           timestamp=timestamp, commit=event_ref.commit_id)


def parse_issue_ref(issue_ref):
    op = Comment(author=issue_ref.user.login, id_='issuecomment_%d' % issue_ref.number,
                 body=issue_ref.body, timestamp=issue_ref.created_at.replace(tzinfo=None))
    issue = Issue(assignee=issue_ref.assignee.login if issue_ref.assignee else None,
                  id_=issue_ref.number, original_post=op, repository=issue_ref.repository.full_name,
                  title=issue_ref.title if issue_ref.title is not None else '')
    replies = list()
    for comment_ref in issue_ref.get_comments():
        replies.append(parse_comment_ref(comment_ref))
    issue.add_replies(replies)
    labels = list()
    for label_ref in issue_ref.get_labels():
        label_ = parse_label_ref(label_ref)
        if label_:
            labels.append(label_)
    issue.add_labels(labels)

    states = list()
    commits = set()
    for event_ref in issue_ref.get_events():
        event = parse_event(event_ref)
        if isinstance(event, StateChange):
            states.append(event)
            if event.commit:
                commits.add(event.commit)
        elif event:
            commits.add(event)
    issue.add_states(states)
    issue.add_commits(commits)
    return issue


if __name__ == '__main__':
    gh = Github()
    repo_ref = gh.get_repo('google/guava')
    pr_ref = repo_ref.get_pull(2825)
    pr = parse_pr_ref(pr_ref, 'google_guava')
    print(pr)
    issue_ref = repo_ref.get_issue(2509)
    issue = parse_issue_ref(issue_ref)
    print(issue)

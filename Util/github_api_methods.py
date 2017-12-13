from datetime import datetime

from github import Github

from gitMine.VCClasses import Comment, Commit, IssueStates, PullRequest


def parse_comment_ref(comment_ref):
    return Comment(id_=comment_ref.id, author=comment_ref.user.name if comment_ref.user else None,
                   body=comment_ref.body, timestamp=comment_ref.created_at)


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
                                                                                    '%a, %d %b %Y %H:%M:%S GMT'),
                  title=title)


def parse_file_ref(file_ref):
    return '%s' % file_ref.filename, '%s' % file_ref.patch


def parse_label_ref(label_ref):
    return label_ref.name if label_ref else None


def parse_pr_ref(pr_ref, project):
    author = pr_ref.user.name if pr_ref.user else None

    comments = [Comment(id_='issuecomment_%d' % pr_ref.number, author=author, body=pr_ref.body,
                        timestamp=pr_ref.created_at)]
    for comment_ref in pr_ref.get_comments():
        comments.append(parse_comment_ref(comment_ref))

    commits = list()
    for commit_ref in pr_ref.get_commits():
        commits.append(parse_commit_ref(commit_ref, project))

    diffs = list()
    for file_ref in pr_ref.get_files():
        diffs.append(parse_file_ref(file_ref))

    labels = list()
    # TODO: Get labels, do we need labels?
    # for label_ref in pr_ref.get_labels():
    #     labels.append(parse_label_ref(label_ref))

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
                       labels=labels, state=state, title=pr_ref.title)


if __name__ == '__main__':
    gh = Github()
    repo_ref = gh.get_repo('google/guava')
    pr_ref = repo_ref.get_pull(2825)
    pr = parse_pr_ref(pr_ref, 'google_guava')

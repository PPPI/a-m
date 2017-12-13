from enum import Enum
from datetime import datetime
from datetime import timedelta
import pytz


class Comment(object):
    def __init__(self, id_, author, body, timestamp):
        self.id_ = id_
        self.author = author
        self.body = body
        self.timestamp = timestamp

    def __cmp__(self, other):
        self.timestamp.__cmp__(other.timestamp)

    def __repr__(self):
        return "Comment: [author=\'" + self.author + "\', body=\'" + self.body + "\', timestamp=\'" \
               + self.timestamp.strftime('%Y-%m-%dT%H:%M:%SZ') + "\']"


class Issue(object):
    def __init__(self, repository, original_post, title, id_, assignee):
        self.original_post = original_post
        self.repository = repository
        self.title = title
        self.id_ = id_
        self.replies = []
        self.commits = []
        self.actions = []
        self.states = []
        self.labels = []
        self.extracted = datetime.now()
        self.assignee = assignee

    def add_replies(self, replies):
        self.replies = replies

    def add_commits(self, commits):
        self.commits = commits

    def add_actions(self, actions):
        self.actions = actions

    def add_labels(self, labels):
        self.labels = labels

    def add_states(self, states):
        self.states = states

    def __repr__(self):
        return 'Issue = [id = %s, title = %s, op = %s, repo = %s, accessed on %s]' % \
               (self.id_, self.title, self.original_post,
                self.repository, self.extracted.strftime('%Y-%m-%d at %H:%M:%S'))


class Commit(object):
    def __init__(self, parent, author, title, desc, branches, repository, c_hash, diff, timestamp):
        self.parent = parent
        self.author = author
        self.title = title
        self.desc = desc
        self.branches = branches
        self.repository = repository
        self.c_hash = c_hash
        self.diff = diff
        self.timestamp = timestamp
        self.extracted = datetime.now()

    def __repr__(self):
        return 'Commit = [author = %s, title = %s, description = %s, hash = %s, diff = %s, extracted on %s]' \
               % (self.author, self.title, self.desc, self.c_hash, self.diff,
                  self.timestamp.strftime('%Y-%m-%d at %H:%M:%S'))

    def is_within_minutes(self, before_min, after_min, target):
        try:
            return (self.timestamp >= target + timedelta(minutes=before_min)) \
                   and (self.timestamp <= target + timedelta(minutes=after_min))
        except TypeError:
            target = pytz.utc.localize(target)
            return (self.timestamp >= target + timedelta(minutes=before_min)) \
                   and (self.timestamp <= target + timedelta(minutes=after_min))


class IssueStates(Enum):
    open = 1
    closed = 2
    merged = 3


class StateChange(object):
    def __init__(self, from_, to_, by, commit, timestamp):
        self.from_ = from_
        self.to_ = to_
        self.by = by
        self.commit = commit
        self.timestamp = timestamp

    def set_commit(self, commit):
        self.commit = commit


class Reference(object):
    def __init__(self, to_, state, by, timestamp):
        self.to_ = to_
        self.state = state
        self.by = by
        self.timestamp = timestamp


class PullRequest(object):
    def __init__(self,
                 number, title, assignee, labels, comments, commits, diffs, state,
                 from_repo, from_branch, to_repo, to_branch):
        self.title = title
        self.number = number
        self.labels = labels
        self.comments = comments
        self.commits = commits
        self.diffs = diffs
        self.state = state
        self.from_repo = from_repo
        self.from_branch = from_branch
        self.to_repo = to_repo
        self.to_branch = to_branch
        self.assignee = assignee


class LineMode(Enum):
    unchanged = 0
    add = 1
    remove = 2
    summary = 3


class FileDiff(object):
    def __init__(self):
        self.lines = []
        self.comments = []

    def add_line(self, original_nr, new_nr, line, mode):
        self.lines.append((original_nr, new_nr, line, mode))

    def add_comment(self, id_, author, body, timestamp, line_index):
        self.comments.append((Comment(id_, author, body, timestamp), line_index))

    def __repr__(self):
        output = ''
        if len(self.comments) > 0:
            i = 0
            j = 0
        for orig_n, nn, line, mode in self.lines:
            if mode == LineMode.summary:
                output += line + '\n'
            else:
                output += orig_n + ' ' + nn + ' '
                output += '+' if mode == LineMode.add else ('-' if mode == LineMode.remove else ' ')
                output += line + '\n'
            if len(self.comments) > 0:
                # noinspection PyUnboundLocalVariable
                comment, index = self.comments[j]
                # noinspection PyUnboundLocalVariable
                if i == index:
                    output += 'Comment by ' + comment.author + ': ' + comment.body + '\nTimestamp: ' + \
                              comment.timestamp.strftime('%Y-%m-%d at %H:%M:%S') + '\n'
                    j = min(j + 1, len(self.comments) - 1)
                i += 1
        return output

    # The following two functions are provided to extract only comments or a proper diff file as needed
    def generate_diff_file(self):
        output = ''
        for orig_n, nn, line, mode in self.lines:
            if mode == LineMode.summary:
                output += ''  # line + '\n'
            else:
                output += '+' if mode == LineMode.add else ('-' if mode == LineMode.remove else ' ')
                output += line + '\n'
        return output

    def extract_comments(self):
        output = ''
        for comment, line_id in self.comments:
            output += str(comment) + ' at line: ' + str(line_id) + '\n'
        return output


class Repository(object):
    def __init__(self, name, langs, commits, issues, prs):
        self.name = name
        self.commits = commits
        self.prs = prs
        self.issues = issues
        self.langs = langs

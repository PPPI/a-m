from random import shuffle
import copy

from gitMine.VCClasses import Repository, Commit, Issue


def generate_leave_one_outs(lolist):
    new_list = list()
    for i in range(len(lolist)):
        bucket = [item for sublist in lolist[:i] + lolist[(i + 1):] for item in sublist]
        new_list.append(bucket)
    return new_list


def cross_split_repo(repo, n_buckets):
    issues = [i for i in repo.issues]
    commits = [c for c in repo.commits]

    links = list()
    for i in issues:
        for c in i.commits:
            links.append((i.id_, c))
    shuffle(links)
    n_buckets = int(len(links) / n_buckets)
    buckets = [links[i:i + n_buckets] for i in range(0, len(links), n_buckets)]

    new_buckets = generate_leave_one_outs(buckets)

    cross_folds = list()
    ids_in_folds = set()
    for batch in new_buckets:
        fold = list()
        for issue_id, commit_id in batch:
            maybe_issue = [i for i in issues if i.id_ == issue_id]
            if len(maybe_issue) != 1:
                continue
            maybe_commit = [c for c in commits if c.c_hash.startswith(commit_id.split('@')[-1])]
            if len(maybe_commit) != 1:
                continue
            issue = maybe_issue[0]
            if issue not in fold:
                new_commits = list()
                for old_commit in issue.commits:
                    if (issue_id, old_commit) in batch:
                        new_commits.append(old_commit)
                issue.commits = new_commits
                fold.append(issue)

            commit = maybe_commit[0]
            if commit not in fold:
                fold.append(commit)

            ids_in_folds.add(issue_id)
            ids_in_folds.add(commit_id)
        cross_folds.append(fold)

    validation_folds = list()
    for batch in buckets:
        fold = list()
        for issue_id, commit_id in batch:
            maybe_issue = [i for i in issues if i.id_ == issue_id]
            if len(maybe_issue) != 1:
                continue
            maybe_commit = [c for c in commits if c.c_hash.startswith(commit_id.split('@')[-1])]
            if len(maybe_commit) != 1:
                continue
            issue = copy.deepcopy(maybe_issue[0])
            if issue not in fold:
                issue.commits = list()
                fold.append(issue)

            commit = maybe_commit[0]
            if commit not in fold:
                fold.append(commit)

        validation_folds.append(fold)

    commits = [c for c in commits if c.c_hash not in ids_in_folds]
    issues = [i for i in issues if i.id_ not in ids_in_folds]
    c_buckets = [commits[i:i + n_buckets] for i in range(0, len(links), n_buckets)]
    new_commits = generate_leave_one_outs(c_buckets)
    i_buckets = [issues[i:i + n_buckets] for i in range(0, len(links), n_buckets)]
    new_issues = generate_leave_one_outs(i_buckets)

    for i in range(len(cross_folds)):
        cross_folds[i] = cross_folds[i] + new_commits[i] + new_issues[i]
        shuffle(cross_folds[i])

    for i in range(len(validation_folds)):
        validation_folds[i] = validation_folds[i] + c_buckets[i] + i_buckets[i]
        shuffle(validation_folds[i])

    training_repos = list(map(lambda fold_: Repository(name=repo.name,
                                                       langs=copy.deepcopy(repo.langs),
                                                       commits=copy.deepcopy([c for c in fold_ if isinstance(c, Commit)]),
                                                       issues=copy.deepcopy([i for i in fold_ if isinstance(i, Issue)]),
                                                       prs=copy.deepcopy(repo.prs)), cross_folds))
    validation_repos = list(map(lambda fold_: Repository(name=copy.deepcopy(repo.name),
                                                         langs=copy.deepcopy(repo.langs),
                                                         commits=copy.deepcopy(
                                                            [c for c in fold_ if isinstance(c, Commit)]),
                                                         issues=copy.deepcopy([i for i in fold_ if isinstance(i, Issue)]),
                                                         prs=copy.deepcopy(repo.prs)), validation_folds))

    links = list()
    for batch in buckets:
        links.append(dict())
        for issue_id, commit_id in batch:
            try:
                links[-1][issue_id].append(commit_id)
            except KeyError:
                links[-1][issue_id] = [commit_id]

    return training_repos, validation_repos, links

import os
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Set
from typing import Tuple
import subprocess
import sys
import shutil

import jsonpickle

from gitMine.VCClasses import Commit

try:
    from urllib.parse import urlparse
except:
    from urlparse import urlparse

__temp_paths__ = []


def clone_git_repo_to_tmp(url: str) -> Tuple[Optional[str], str]:
    parsed_url = urlparse(url)
    if parsed_url.scheme == "file" or parsed_url.scheme == "git" \
            or parsed_url.scheme == "http" or parsed_url.scheme == "https" \
            or parsed_url.scheme == "ssh":
        path = tempfile.mkdtemp(suffix=".gitScraper")

        git_clone = subprocess.Popen(["git", "clone", url, path], bufsize=1, stdout=sys.stderr)
        git_clone.wait()

        if git_clone.returncode != 0:
            sys.exit(git_clone.returncode)

        __temp_paths__.append(path)
        return os.path.basename(parsed_url.path), path

    return None, os.path.abspath(url)


def clean_up():
    for path in __temp_paths__:
        shutil.rmtree(path, ignore_errors=True)


def get_all_commit_hashes(path: str) -> List[str]:
    commit_hashes_process = subprocess.Popen(['git', 'log', '--branches=*', '--format=oneline'], bufsize=1,
                                             stdout=subprocess.PIPE, cwd=path).stdout
    commit_hashes = commit_hashes_process.readlines()
    commit_hashes_process.close()
    # TODO: Validate that this provides all hashes for ALL branches, i.e. dead-ends, and all tree paths
    # XXX: Well, there are cases where branches go missing
    return list(map(lambda line: line.decode('utf-8', 'replace').split(' ')[0].strip(), commit_hashes))


def process_a_commit(hash_: str, name: str, path: str) -> Commit:
    commit_show_process = subprocess.Popen(['git', 'show', '--format=fuller', hash_], bufsize=1, stdout=subprocess.PIPE,
                                           cwd=path).stdout
    show_lines = list(map(lambda line: line.decode('utf-8', 'replace'), commit_show_process.readlines()))
    commit_show_process.close()

    # Sanity check that we are looking at the expected commit
    assert (show_lines[0].split(' ')[-1].strip() == hash_)

    current_index = 0
    while not (show_lines[current_index].startswith('Author')):
        current_index += 1
    author = show_lines[current_index].split(':')[-1][1:].strip().split('<')[0][:-1]
    # We want to remove 'CommitDate: ' from the start, hence 12
    timestamp = datetime.strptime(show_lines[current_index + 3][12:].strip(), '%a %b %d %H:%M:%S %Y %z')

    current_index += 4
    while not (show_lines[current_index] == '\n'):
        current_index += 1
    current_index += 1

    title = ''
    if current_index < len(show_lines):
        while True:
            curr_line = show_lines[current_index]
            if curr_line == '\n':
                current_index += 1
                break
            title += curr_line[4:]
            current_index += 1
            if current_index == len(show_lines):  # We reached EOF in title, i.e. the commit has no diff-s
                break
        title = title.strip()

    diffs = []
    if current_index < len(show_lines):
        current_diff = ''
        while True:
            curr_line = show_lines[current_index]
            if current_index + 1 == len(show_lines):  # We reached the EOF
                current_diff += curr_line
                diffs.append(current_diff.strip())
                break
            elif curr_line.startswith('diff --git a'):  # New diff for this commit
                if current_diff != '':
                    diffs.append(current_diff.strip())
                current_diff = curr_line
            else:
                current_diff += curr_line
            current_index += 1

    # Now we request the branches
    brach_cmd = subprocess.Popen(['git', 'branch', '-r', '--contains'], bufsize=1, stdout=subprocess.PIPE,
                                 cwd=path).stdout
    branches = set(map(lambda line: line.decode('utf-8', 'replace').strip().split(' ')[-1], brach_cmd.readlines()))
    brach_cmd.close()

    # And unless told this is the first commit, the parent
    parent = ''
    commit_show_process = subprocess.Popen(['git', 'show', '--format=oneline', hash_ + '^'], bufsize=1,
                                           stdout=subprocess.PIPE, cwd=path).stdout
    show_lines = commit_show_process.readlines()
    commit_show_process.close()
    try:
        parent = show_lines[0].decode('utf-8', 'replace').split(' ')[0]
    except IndexError:
        pass  # The commit does not have a Parent

    return Commit(author=author, branches=branches, c_hash=hash_, diff=diffs, desc=title, parent=parent,
                  repository=name, timestamp=timestamp, title=title.split('\n')[0])


def generate_name_nick_map(hash_: str, path: str, map_so_far: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    commit_show_process = subprocess.Popen(['git', 'show', '--format=fuller', hash_], bufsize=1,
                                           stdout=subprocess.PIPE,
                                           cwd=path).stdout
    show_lines = list(map(lambda line: line.decode('utf-8', 'replace'), commit_show_process.readlines()))
    commit_show_process.close()

    # Sanity check that we are looking at the expected commit
    assert (show_lines[0].split(' ')[-1].strip() == hash_)

    current_index = 0
    while not (show_lines[current_index].startswith('Author')):
        current_index += 1
    author = show_lines[current_index].split(':')[-1][1:].strip().split('<')[0][:-1]
    nick = show_lines[current_index].split(':')[-1][1:].strip().split('<')[1][:-1]
    try:
        map_so_far[author].add(nick)
    except KeyError:
        map_so_far[author] = {nick}
    return map_so_far


if __name__ == '__main__':
    try:
        repo_url = 'http://git-wip-us.apache.org/repos/asf/commons-lang.git'
        repo_name, repo_path = clone_git_repo_to_tmp(repo_url)
        hashes = get_all_commit_hashes(repo_path)
        commits = list(map(lambda h: process_a_commit(h, repo_name, repo_path), hashes))
        # map_so_far = dict()
        # for h in hashes:
        #     map_so_far = generate_name_nick_map(h, repo_path, map_so_far)
        # print(map_so_far)
        filename = repo_name + '.json'
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'Apache Git', filename), 'w',
                  encoding='utf-8') as f:
            f.write(jsonpickle.encode(commits))
    finally:
        # TODO: Verify python finally behaviour, is this safe for all spare .kill() ? Can we assume the OS will clean ?
        clean_up()

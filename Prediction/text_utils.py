from typing import List, Set, Union

from gensim.parsing import PorterStemmer
from gensim.utils import tokenize

from Util import utils_
from gitMine.VCClasses import Commit, Issue, PullRequest

ps = PorterStemmer()


def merge_commit_title_and_desc(commit: Commit) -> str:
    if commit.title[-3:] == commit.desc[:3] == '...':
        return commit.title[:-3] + commit.desc[3:]
    return commit.title + commit.desc


def preprocess_text(text: str, stopwords_: Set[str], min_len) -> List[str]:
    return [ps.stem(tok).lower()
            for tok in split_compound_toks(tokenize(text, lowercase=False, deacc=True, errors='strict'))
            if tok not in stopwords_ and len(tok) > min_len]


def split_compound_toks(toks: List[str]) -> List[str]:
    """Method to split CamelCase and snake_case by first normalizing everything to snake_case and splitting on '_'"""
    result = list()
    for tok in toks:
        candidate = utils_.GitMineUtils.camel_to_snake(tok).translate({ord(c): ' ' for c in '_'})
        if not tok == candidate:
            result += [tok_ for tok_ in candidate.split(' ')]
        result.append(''.join(tok.split('_')))
    return result


def text_pipeline(tokenizable: Union[Issue, Commit, PullRequest], stopwords_: Set[str], min_len) -> List[str]:
    if isinstance(tokenizable, Issue):
        text = [tokenizable.title] \
               + ([tokenizable.original_post.body]
                  if (tokenizable.original_post.body != 'No description provided.'
                      and tokenizable.original_post.body is not None) else []) \
               + [comm.body for comm in tokenizable.replies]
    elif isinstance(tokenizable, Commit):
        text = [merge_commit_title_and_desc(tokenizable)] + ([tokenizable.diff] if isinstance(tokenizable.diff, str)
                                                             else tokenizable.diff)
    elif isinstance(tokenizable, PullRequest):
        text = [_ for sublist in
                [
                    [merge_commit_title_and_desc(commit)]
                    + [diff[1] if isinstance(diff[1], str) else diff[1].generate_diff_file() for diff in commit.diff]
                    for commit in tokenizable.commits
                ]
                for _ in sublist] \
               + [tokenizable.comments[0].body] if tokenizable.comments else [] + [tokenizable.title]
    else:
        raise ValueError('Unexpected object type, tokenizable must be either an Issue, Commit or PullRequest')
    text = '\n'.join([t if t is not None else '' for t in text])
    return preprocess_text(text, stopwords_, min_len)

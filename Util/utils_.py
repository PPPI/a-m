import os
import re
from fnmatch import fnmatch
from typing import List

import numpy
import numpy as np


class GitMineUtils:
    DELTA = 900

    STOPWORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
                 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
                 'can\'t',
                 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down',
                 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
                 'haven\'t',
                 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him',
                 'himself',
                 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t',
                 'it',
                 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor',
                 'not',
                 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours 	ourselves', 'out', 'over',
                 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some',
                 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
                 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through',
                 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re',
                 'we\'ve',
                 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who',
                 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
                 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves', '']

    FIRST_CAP_RE = re.compile(r'(.)([A-Z][a-z]+)', flags=re.UNICODE)
    ALL_CAP_RE = re.compile(r'([a-z0-9])([A-Z])', flags=re.UNICODE)
    SCD_SEP_RE = re.compile(r'(\w)(\*|::?|\.|)(\w)', flags=re.UNICODE)

    # THIS IS HERE TO AVOID SPLITTING URLs, hoping nobody uses IPv6 directly!!!
    IS_URL_LIKE = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', flags=re.IGNORECASE & re.UNICODE)

    @staticmethod
    def get_pattern_paths(pattern: str, loc: str=os.path.join(os.getcwd(), '..', 'data')) -> List[str]:
        """
        Find the OS paths to all files that match a pattern
        :param pattern: The regEx pattern to match the filename to
        :param loc: The directory in which to look for the pattern, defaults to the parent of the working directory
        :return: A list of paths to the found files, empty list when no files found
        """
        files_paths = []
        for path, subdirs, files in os.walk(loc):
            for name in files:
                if fnmatch(name, pattern):
                    files_paths.append(os.path.join(path, name))
        return files_paths

    @staticmethod
    def is_outlier(points: numpy.array, thresh=3.5) -> numpy.array:
        """
        Returns a boolean array with True if points are outliers and False
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    @staticmethod
    def camel_to_snake(name: str) -> str:
        s1 = GitMineUtils.FIRST_CAP_RE.sub(r'\1_\2', name)
        return GitMineUtils.ALL_CAP_RE.sub(r'\1_\2', s1)

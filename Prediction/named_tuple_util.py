from collections import namedtuple

link_datapoint = namedtuple('link_datapoint', 'issue pr linked in_comments is_reporter '
                                              'is_assignee engagement lag '
                                              'jaccard files top_2 cosine cosine_tt '
                                              'cosine_tc cosine_ct cosine_cc lag_open lag_close '
                                              'no_pr_desc pr_commits pr_files')

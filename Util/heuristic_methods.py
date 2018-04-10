import re


def extract_issue_numbers(text):
    return re.findall(r'#[0-9]+\b', text)


def extract_commit_shas(text):
    return re.findall(r'[a-f0-9]{5,32}\b', text)
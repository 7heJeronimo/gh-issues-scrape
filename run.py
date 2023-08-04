import time
import logging
import argparse
from datetime import datetime
from typing import List

from tqdm import tqdm
from github import Auth, Github
from github.GithubException import RateLimitExceededException
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract GitHub repository issues.")
    parser.add_argument("--org", help="GitHub organization name containing the repositories.")
    parser.add_argument("--repos", nargs="+", help="List of GitHub repository names within the organization.")
    parser.add_argument("--gh_token", help="GitHub personal access token for authentication.")
    parser.add_argument("--hf_token", help="Hugging Face token for pushing the dataset to the model hub.")
    parser.add_argument("--hf_repo", help="Hugging Face org/repo where the dataset should be uploaded to." )
    parser.add_argument("--ds_path", help="Local path where dataset progress is saved.")
    return parser.parse_args()


def extract_gh_repo_issues(
    gh_org: str, gh_repos: List[str], gh_token: str, ds_path: str
):
    """
    Extracts and collects information about issues from multiple GitHub repositories.

    Parameters:
        gh_org (str): The GitHub organization name containing the repositories.
        gh_repos (List[str]): A list of GitHub repository names within the organization.
        gh_token (str): The GitHub personal access token for authentication.
        ds_path (str): Local path where dataset progress is saved

    Returns:
        Dataset: A Hugging Face Dataset object containing the extracted issue information.

    Description:
        This function collects issue data from multiple repositories within the specified GitHub
        organization. It utilizes the GitHub API with the provided personal access token for
        authentication.

        The function goes through each repository in the `gh_repos` list, fetches the list of
        issues from each repository (both open and closed), and extracts relevant information
        about each issue. The extracted information includes the repository name, issue number,
        state (open or closed), issue title, issue body, creation date, closure date (if closed),
        and comments (concatenated with a special delimiter).

        The function employs a loop to handle potential GitHub API rate limits. If the rate limit
        is exceeded during data extraction, the function pauses and waits until the rate limit is
        reset before continuing.

        Finally, the collected issue data is transformed into a Hugging Face Dataset object and
        returned.
    """
    all_issues = []

    for repo_name in gh_repos:
        auth = Auth.Token(gh_token)
        gh = Github(auth=auth)

        logging.info(f"Collecting issues for repo: {repo_name}")

        repo = gh.get_repo(f"{gh_org}/{repo_name}")
        gh_issues = repo.get_issues(state="all")
        num_issues = gh_issues.totalCount

        # collect issues
        for i, issue in enumerate(tqdm(gh_issues, total=num_issues)):
            try:
                comments = "<|||||>".join(
                    [comment.body for comment in issue.get_comments()]
                )  # weird delimiter to avoid conflicts with common existing delimiters

                issue_data = {
                    "repo": repo_name,
                    "number": issue.number,
                    "state": issue.state,
                    "title": issue.title,
                    "body": issue.body,
                    "created_at": issue.created_at.strftime("%m-%d-%Y %H:%M:%S"),
                    "closed_at": issue.created_at.strftime("%m-%d-%Y %H:%M:%S"),
                    "comments": comments,
                }

                all_issues.append(issue_data)

            except RateLimitExceededException as e:
                logging.info(f"Rate limit exceeded at {i} of {num_issues} total issues")

                # save progress
                dataset = Dataset.from_list(all_issues)
                dataset.save_to_disk(ds_path)

                rate_limit = gh.get_rate_limit()
                wait_time = (rate_limit.raw_data["core"]["reset"] - time.time()) + 3
                logging.info(
                    f'Current time is {datetime.now().strftime("%m-%d-%Y %H:%M:%S")}\n',
                    f"Waiting {wait_time/60} minutes until rate limit is reset.\n\n",
                )
                time.sleep(wait_time)

    dataset = Dataset.from_list(all_issues)

    return dataset


if __name__ == "__main__":

    args = parse_args()
    
    logging.info(f'{"-"*5} Start {"-"*5}')
    dataset = extract_gh_repo_issues(gh_org=args.org, gh_repos=args.repos, gh_token=args.gh_token, ds_path="./hf_repo_issues")
    dataset.push_to_hub(repo_id=args.hf_repo, token=args.hf_token)
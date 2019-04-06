import praw

reddit = praw.Reddit()

for comment in reddit.subreddit('rit').hot(limit=5):

	print(vars(comment))
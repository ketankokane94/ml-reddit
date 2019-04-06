import praw
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID', client_secret="YOUR_SECRET",
                     password='YOUR_PASSWORD', user_agent='USERNAME',
                     username='USERNAME')

for comment in reddit.subreddit('rit').stream.comments():
	print(vars(comment))
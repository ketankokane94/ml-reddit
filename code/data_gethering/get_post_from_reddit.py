from data_gethering.connect_to_reddit import connect_to_reddit
import json

data = []
reddit = connect_to_reddit()

subredditName = 'IAmA'
for submission in reddit.subreddit(subredditName).top(limit = 2000):
    data.append({'body': submission.selftext,  'class': subredditName})

# reddit = connect_to_reddit()
subredditName = 'stories'
for submission in reddit.subreddit(subredditName).top(limit = 2000):
    data.append({'body': submission.selftext,  'class': subredditName})


with open('posts.json', 'w') as fp:
    json.dump(data, fp)
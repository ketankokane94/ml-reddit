from data_gethering.connect_to_reddit import connect_to_reddit
import json

data = []
reddit = connect_to_reddit()
for comment in reddit.subreddit('iama').hot(limit=5):
    data.append({'seltext' :comment.selftext, 'title': comment.title,'subredditName': 'iama' })
print(data)


with open('posts.json', 'w') as fp:
    json.dump(data, fp)

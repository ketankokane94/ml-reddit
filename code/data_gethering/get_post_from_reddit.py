from data_gethering.connect_to_reddit import connect_to_reddit
import json

data = []
reddit = connect_to_reddit()
i=0

for submission in reddit.subreddit('documentaries').top(limit = 2000):
    print(i)
    i += 1
    print(vars(submission))
    data.append({'seltext': submission.selftext, 'title': submission.title, 'subredditName':
        'documentaries'})

reddit = connect_to_reddit()

for submission in reddit.subreddit('lifehacks').top(limit = 2000):
    print(i)
    i += 1
    print(vars(submission))
    data.append({'seltext': submission.selftext, 'title': submission.title, 'subredditName': 'lifehacks'})


with open('posts.json', 'w') as fp:
    json.dump(data, fp)
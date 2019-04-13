import praw

#passwords and usernames have been removed for obvious reasons

def connect_to_reddit():
    reddit = praw.Reddit(client_id='xxxxxx', client_secret="xxx-xxxxx",
                         password='xxxxx', user_agent='xxxx',
                         username='xxxx')
    return reddit


if __name__ == '__main__':
    # add testing code here
    pass

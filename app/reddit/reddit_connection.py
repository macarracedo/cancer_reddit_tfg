import json

import praw

'''
Como recoger el path del fichero data?
path = Path.cwd()
print(f'{str(path)}')
'''

# Usando ruta absoluta
credentials = '/home/manuel/Documentos/cancer_reddit_tfg/data/client_secrets.json'

with open(credentials) as f:
    creds = json.load(f)

# Config Reddit connection
reddit = praw.Reddit(client_id=creds['client_id'],
                     client_secret=creds['client_secret'],
                     user_agent=creds['user_agent'],
                     redirect_uri=creds['redirect_uri'],
                     refresh_token=creds['refresh_token'])
reddit.validate_on_submit = True

enable_commit = True
enable_submission_insert = True
enable_comment_insert = True
enable_redditor_insert = True
post_limit = 50000
comment_depth_limit = 4

subreddit = reddit.subreddit("cancer")
listing = subreddit.new(limit=post_limit)

import praw
import json

my_id = 'cVkuW1BdyZ-eCg'
my_secret = 'wAl9oV4E-HsecgwuW1Lj1sCY7kA'
my_agent = 'Python:AdviceBot:v2 (by /u/terrie9876)'

reddit = praw.Reddit(client_id=my_id,
                     client_secret=my_secret,
                     user_agent=my_agent)
print(reddit.read_only)
json_contents = {'submissions': []}


def get_best_comments(submission):
    submission.comment_sort = 'best'
    ans = []
    for top_level_comment in submission.comments:
        if len(ans) >= 5:
            break
        if isinstance(top_level_comment, praw.models.MoreComments) or top_level_comment.body.lower() == '[deleted]' or top_level_comment.author == submission.author:
            continue
        ans.append(top_level_comment.body)
    return ans


count = 0
for submission in reddit.subreddit('relationships').top(limit=1000):
    if count % 50 == 0:
        print('At count ' + str(count))
    if submission.link_flair_text != 'Updates' and submission.title.lower().find('update') == -1 and submission.num_comments > 5:
        sub_id = submission.id
        sub_contents = submission.selftext
        sub_comments = get_best_comments(submission)
        if len(sub_comments) != 5:
            continue
        sub_edited = int(submission.edited)
#       Storing only the information that I need to work with
        json_contents['submissions'].append({
           'id' : sub_id,
           'body' : sub_contents,
           'comments' : sub_comments,
            'edited' : sub_edited > 0
           })
        count += 1

print(count)

with open('Raw Data/data.json', 'w') as fp:
    fp.seek(0)
    fp.truncate()
    json.dump(json_contents, fp)


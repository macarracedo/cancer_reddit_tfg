from db.db_connection import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey

from .redditor_table import Redditor
from .subreddit_table import Subreddit


class Submission(Base):
    __tablename__ = 'submissions'
    allow_live_comments = Column(Boolean)
    # approved_at_utc = {NoneType}
    # approved_by = {NoneType}
    archived = Column(Boolean)
    fk_id_author = Column(Integer, ForeignKey(Redditor.id))
    id_author = Column(String)
    # author_flair_background_color = {NoneType}
    # author_flair_css_class = {NoneType}
    # author_flair_richtext = {list: 0}[]
    # author_flair_template_id = {NoneType}
    # author_flair_text = {NoneType}
    # author_flair_text_color = {NoneType}
    author_flair_type = Column(String)
    author_fullname = Column(String)
    author_is_blocked = Column(Boolean)
    author_patreon_flair = Column(Boolean)
    author_premium = Column(Boolean)
    # awarders = {list: 0}[]
    # banned_at_utc = {NoneType}
    # banned_by = {NoneType}
    can_gild = Column(Boolean)
    can_mod_post = Column(Boolean)
    # category = {NoneType}
    clicked = Column(Boolean)
    comment_limit = Column(Integer)
    comment_sort = Column(String)
    # comments = {CommentForest: 55}
    # content_categories = {NoneType}
    contest_mode = Column(Boolean)
    created = Column(Float)
    created_utc = Column(Float)
    # discussion_type = {NoneType}
    # distinguished = {NoneType}
    domain = Column(String)
    downs = Column(Integer)
    edited = Column(Float)
    # flair = {SubmissionFlair}
    fullname = Column(String)
    gilded = Column(Integer)
    # gildings = {dict: 1}
    hidden = Column(Boolean)
    hide_score = Column(Boolean)
    id = Column(Integer, primary_key=True, autoincrement=True)
    id_submission = Column(String, unique=True)  # Id asignado por reddit
    is_created_from_ads_ui = Column(Boolean)
    is_crosspostable = Column(Boolean)
    is_meta = Column(Boolean)
    is_original_content = Column(Boolean)
    is_reddit_media_domain = Column(Boolean)
    is_robot_indexable = Column(Boolean)
    is_self = Column(Boolean)
    is_video = Column(Boolean)
    # likes = {NoneType}
    link_flair_background_color = Column(String)
    link_flair_css_class = Column(String)
    # link_flair_richtext = {list: 0}[]
    # link_flair_text = {NoneType}
    link_flair_text = Column(String)
    link_flair_type = Column(String)
    locked = Column(Boolean)
    # media = {NoneType}
    # media_embed = {dict: 0}
    media_only = Column(Boolean)
    # mod = {SubmissionModeration}
    # mod_note = {NoneType}
    # mod_reason_by = {NoneType}
    # mod_reason_title = {NoneType}
    # mod_reports = {list: 0}[]
    name = Column(String)
    no_follow = Column(Boolean)
    num_comments = Column(Integer)
    num_crossposts = Column(Integer)
    num_duplicates = Column(Integer)
    # num_reports = {NoneType}
    over_18 = Column(Boolean)
    parent_whitelist_status = Column(String)
    permalink = Column(String)
    pinned = Column(Boolean)
    pwls = Column(Integer)
    quarantine = Column(Boolean)
    # removal_reason = {NoneType}
    # removed_by = {NoneType}
    # removed_by_category = {NoneType}
    # report_reasons = {NoneType}
    saved = Column(Boolean)
    score = Column(Integer)
    # secure_media = {NoneType}
    #  secure_media_embed = {dict: 0}
    selftext = Column(String)
    selftext_html = Column(String)
    send_replies = Column(Boolean)
    shortlink = Column(String)
    spoiler = Column(Boolean)
    stickied = Column(Boolean)
    # subreddit = {Subreddit}
    subreddit_id = Column(String, ForeignKey(Subreddit.name))
    subreddit_name_prefixed = Column(String)
    subreddit_subscribers = Column(Integer)
    subreddit_type = Column(String)
    # suggested_sort = {NoneType}
    # thumbnail = Column(String)
    # thumbnail_height = {NoneType}
    # thumbnail_width = {NoneType}
    title = Column(String)
    # top_awarded_type = {NoneType}
    total_awards_received = Column(Integer)
    # treatment_tags = {list: 0}
    ups = Column(Integer)
    upvote_ratio = Column(Float)
    url = Column(String)
    # user_reports = {list: 0}
    # view_count = {NoneType}
    visited = Column(Boolean)
    whitelist_status = Column(String)
    wls = Column(Integer)

    def __int__(self, allow_live_comments, archived, fk_id_author, id_author, author_flair_type, author_fullname,
                author_is_blocked,
                author_patreon_flair, author_premium, can_gild, can_mod_post, clicked, comment_limit, comment_sort,
                contest_mode, created, created_utc, domain, downs, edited, fullname, gilded, hidden, hide_score, id,
                id_submission, is_created_from_ads_ui, is_crosspostable, is_meta, is_original_content,
                is_reddit_media_domain, is_robot_indexable, is_self, is_video, link_flair_background_color,
                link_flair_css_class, link_flair_text_color, link_flair_type, locked, media_only, name, no_follow,
                num_comments, num_crossposts, num_duplicates, over_18, parent_whitelist_status, permalink, pinned, pwls,
                quarantine, saved, score, selftext, selftext_html, send_replies, shortlink, spoiler, stickied,
                subreddit_id, subreddit_name_prefixed, subreddit_subscribers, subreddit_type, title,
                total_awards_received,
                ups, upvote_ratio, url, visited, whitelist_status, wls, flair):
        self.allow_live_comments = allow_live_comments
        self.archived = archived
        self.fk_id_author = fk_id_author
        self.id_author = id_author
        self.author_flair_type = author_flair_type
        self.author_fullname = author_fullname
        self.author_is_blocked = author_is_blocked
        self.author_patreon_flair = author_patreon_flair
        self.author_premium = author_premium
        self.can_gild = can_gild
        self.can_mod_post = can_mod_post
        self.clicked = clicked
        self.comment_limit = comment_limit
        self.comment_sort = comment_sort
        self.contest_mode = contest_mode
        self.created = created
        self.created_utc = created_utc
        self.domain = domain
        self.downs = downs
        self.edited = edited
        self.fullname = fullname
        self.flair = flair
        self.gilded = gilded
        self.hidden = hidden
        self.hide_score = hide_score
        self.id = id
        self.id_submission = id_submission
        self.is_created_from_ads_ui = is_created_from_ads_ui
        self.is_crosspostable = is_crosspostable
        self.is_meta = is_meta
        self.is_original_content = is_original_content
        self.is_reddit_media_domain = is_reddit_media_domain
        self.is_robot_indexable = is_robot_indexable
        self.is_self = is_self
        self.is_video = is_video
        self.link_flair_background_color = link_flair_background_color
        self.link_flair_css_class = link_flair_css_class
        self.link_flair_text_color = link_flair_type
        self.link_flair_type = link_flair_type
        self.locked = locked
        self.media_only = media_only
        self.name = name
        self.no_follow = no_follow
        self.num_comments = num_comments
        self.num_crossposts = num_crossposts
        self.num_duplicates = num_duplicates
        self.over_18 = over_18
        self.parent_whitelist_status = parent_whitelist_status
        self.permalink = permalink
        self.pinned = pinned
        self.pwls = pwls
        self.quarantine = quarantine
        self.saved = saved
        self.score = score
        self.selftext = selftext
        self.selftext_html = selftext_html
        self.send_replies = send_replies
        self.shortlink = shortlink
        self.spoiler = spoiler
        self.stickied = stickied
        self.subreddit_id = subreddit_id
        self.subreddit_name_prefixed = subreddit_name_prefixed
        self.subreddit_subscribers = subreddit_subscribers
        self.subreddit_type = subreddit_type
        self.title = title
        self.total_awards_received = total_awards_received
        self.ups = ups
        self.upvote_ratio = upvote_ratio
        self.url = url
        self.visited = visited
        self.whitelist_status = whitelist_status
        self.wls = wls

    '''
    def __init__(self, id_submission, title, selftext, fk_id_author, id_author, fk_name_subreddit, ups, downs,
                 upvote_ratio, url, ):
        self.id_submission = id_submission
        self.title = title
        self.selftext = selftext
        self.fk_id_author = fk_id_author
        self.id_author = id_author
        self.fk_name_subreddit = fk_name_subreddit
        self.ups = ups
        self.downs = downs
        self.upvote_ratio = upvote_ratio
        self.url = url
    '''

    def __repr__(self):
        return "id=%d id_submission=%s title=%s sefltext=%s fk_id_author=%d id_author=%s ups=%d downs=%d upvote_ratio=%f url=%s" \
               % (
                   self.id, self.id_submission, self.title, self.selftext, self.fk_id_author, self.id_author,
                   self.ups, self.downs, self.upvote_ratio, self.url)

    def __str__(self):
        return "id=%d id_submission=%s title=%s sefltext=%s fk_id_author=%d id_author=%s ups=%d downs=%d upvote_ratio=%f url=%s" \
               % (
                   self.id, self.id_submission, self.title, self.selftext, self.fk_id_author, self.id_author,
                   self.ups, self.downs, self.upvote_ratio, self.url)

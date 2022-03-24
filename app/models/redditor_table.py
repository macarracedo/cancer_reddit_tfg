from db.db_connection import Base
from sqlalchemy import Column, Integer, String, Float, Boolean


class Redditor(Base):
    __tablename__ = 'redditors'
    STR_FIELD = Column(String)
    accept_chats = Column(Boolean)
    accept_followers = Column(Boolean)
    accept_pms = Column(Boolean)
    awardee_karma = Column(Integer)
    awarder_karma = Column(Integer)
    comment_karma = Column(Integer)
    created = Column(Float)
    created_utc = Column(Float)
    fullname = Column(String)
    has_subscribed = Column(Boolean)
    has_verified_email = Column(Boolean)
    hide_from_robots = Column(Boolean)
    icon_img = Column(String)
    id = Column(Integer, primary_key=True)
    id_redditor = Column(String, unique=True, nullable=False)
    is_blocked = Column(Boolean)
    is_employee = Column(Boolean)
    is_friend = Column(Boolean)
    is_gold = Column(Boolean)
    is_mod = Column(Boolean)
    link_karma = Column(Integer)
    name = Column(String, unique=True, nullable=False)  # unique because it's an username
    pref_show_snoovatar = Column(Boolean)
    snoovatar_img = Column(String)
    total_karma = Column(Integer)
    verified = Column(Boolean)

    def __init__(self, id_redditor, name, total_karma=None, link_karma=None, comment_karma=None, awardee_karma=None,
                 awarder_karma=None,
                 created=None, created_utc=None, icon_img_url=None, verified=None, is_blocked=None, is_suspended=None,
                 is_employee=None, is_friend=None, is_mod=None, is_gold=None,
                 accept_chats=None, accept_followers=None, accept_pms=None, has_verified_email=None,
                 has_subscribed=None, hide_from_robots=None):
        self.id_redditor = id_redditor
        self.name = name
        self.total_karma = total_karma
        self.link_karma = link_karma
        self.comment_karma = comment_karma
        self.awardee_karma = awardee_karma
        self.awarder_karma = awarder_karma
        self.created = created
        self.created_utc = created_utc
        self.icon_img_url = icon_img_url
        self.verified = verified
        self.is_blocked = is_blocked
        self.is_suspended = is_suspended
        self.is_employee = is_employee
        self.is_friend = is_friend
        self.is_mod = is_mod
        self.is_gold = is_gold
        self.accept_chats = accept_chats
        self.accept_followers = accept_followers
        self.accept_pms = accept_pms
        self.has_verified_email = has_verified_email
        self.has_subscribed = has_subscribed
        self.hide_from_robots = hide_from_robots

    def __repr__(self):
        return f'id={self.id},id_redditor={self.id_redditor}, name={self.name}, ' \
               f'total_karma={self.total_karma}, link_karma={self.link_karma}, comment_karma={self.comment_karma},' \
               f'awardee_karma={self.awardee_karma}, awarder_karma={self.awarder_karma}, created={self.created}, ' \
               f'created_utc={self.created_utc}, icon_img_url={self.icon_img_url}, verified={self.verified}, ' \
               f'is_blocked={self.is_blocked}, is_suspended={self.is_suspended}, is_employee={self.is_employee},' \
               f'is_friend={self.is_friend}, is_mod={self.is_mod}, is_gold={self.is_gold}, accept_chats={self.accept_chats},' \
               f'accept_followers={self.accept_followers}, accept_pms={self.accept_pms}, has_verified_email={self.has_verified_email},' \
               f'has_subscribed={self.has_subscribed}, hide_from_robots={self.hide_from_robots}'

    def __str__(self):
        return f'id={self.id},id_redditor={self.id_redditor}, name={self.name}, ' \
               f'total_karma={self.total_karma}, link_karma={self.link_karma}, comment_karma={self.comment_karma},' \
               f'awardee_karma={self.awardee_karma}, awarder_karma={self.awarder_karma}, created={self.created}, ' \
               f'created_utc={self.created_utc}, icon_img_url={self.icon_img_url}, verified={self.verified}, ' \
               f'is_blocked={self.is_blocked}, is_suspended={self.is_suspended}, is_employee={self.is_employee},' \
               f'is_friend={self.is_friend}, is_mod={self.is_mod}, is_gold={self.is_gold}, accept_chats={self.accept_chats},' \
               f'accept_followers={self.accept_followers}, accept_pms={self.accept_pms}, has_verified_email={self.has_verified_email},' \
               f'has_subscribed={self.has_subscribed}, hide_from_robots={self.hide_from_robots}'

from db.db_connection import Base
from sqlalchemy import Column, Integer, String, ForeignKey

from .redditor_table import Redditor
from .submission_table import Submission


class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True)
    id_comment = Column(String)  # ID que asigna reddit
    fk_id_submission = Column(Integer, ForeignKey(Submission.id))
    id_submission = Column(String)  # ID que asigna reddit
    fk_id_author = Column(Integer, ForeignKey(Redditor.id))
    id_author = Column(String)  # ID que asigna reddit
    id_parent = Column(String)
    body = Column(String)
    ups = Column(Integer)
    downs = Column(Integer)
    depth = Column(Integer)

    def __init__(self, id_comment, fk_id_submission, id_submission, fk_id_author, id_author, id_parent, body, ups,
                 downs, depth):
        self.id_comment = id_comment
        self.fk_id_submission = fk_id_submission
        self.id_submission = id_submission
        self.fk_id_author = fk_id_author
        self.id_author = id_author
        self.id_parent = id_parent
        self.body = body
        self.ups = ups
        self.downs = downs
        self.depth = depth

    def __repr__(self):
        return "id=%d id_comment=%s fk_id_submission=%d id_submission=%s fk_id_author=%d id_author=%s fk_id_parent=%d id_parent=%s body=%s ups=%d downs=%d depth=%d" \
               % (self.id, self.id_comment, self.fk_id_submission, self.id_submission, self.fk_id_author,
                  self.id_author, self.id_parent, self.body, self.ups, self.downs, self.depth)

    def __str__(self):
        return "id=%d id_comment=%s fk_id_submission=%d id_submission=%s fk_id_author=%d id_author=%s fk_id_parent=%d id_parent=%s body=%s ups=%d downs=%d depth=%d" \
               % (self.id, self.id_comment, self.fk_id_submission, self.id_submission, self.fk_id_author,
                  self.id_author, self.id_parent, self.body, self.ups, self.downs, self.depth)

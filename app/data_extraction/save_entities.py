from sqlalchemy import select


def saveSubreddit(subreddit):
    db_subreddit = select(Subreddit.name).where(subreddit.display_name == Subreddit.name)
    resultado = db.session.execute(db_subreddit).fetchone()
    if resultado is not None:  # conocemos al subreddit
        print(
            "*********** NO insertado el \"Subreddit\" con display_name: " + subreddit.display_name + " *********** (Ya en la BD)")
        db_subreddit = None
    else:  # no tenemos este subreddit -> lo anhadimos
        db_subreddit = Subreddit(name=subreddit.display_name)
    return db_subreddit


def saveRedditor(redditor):
    if redditor is not None:
        if hasattr(redditor, 'id'):  # Tiene atributo 'id'
            query = select(Redditor.id_redditor).where(Redditor.id_redditor == redditor.id)
            result = db.session.execute(query).fetchone()  # devuelve uno
            if result is not None:  # conocemos al redditor
                # Sustituir por logger
                print("*********** NO insertado el \"Redditor\" con id: " + redditor.id + " *********** (Ya en la BD)")
                db_author = None
            else:  # no tenemos este redditor -> lo anhadimos
                db_author = Redditor(id_redditor=redditor.id, name=redditor.name,
                                     total_karma=redditor.total_karma, link_karma=redditor.link_karma,
                                     comment_karma=redditor.comment_karma, awardee_karma=redditor.awardee_karma,
                                     awarder_karma=redditor.awarder_karma, created=redditor.created,
                                     created_utc=redditor.created_utc, icon_img_url=redditor.icon_img,
                                     verified=redditor.verified,
                                     is_blocked=redditor.is_blocked, is_employee=redditor.is_employee,
                                     is_friend=redditor.is_friend,
                                     is_mod=redditor.is_mod, is_gold=redditor.is_gold,
                                     accept_chats=redditor.accept_chats,
                                     accept_followers=redditor.accept_followers, accept_pms=redditor.accept_pms,
                                     has_verified_email=redditor.has_verified_email,
                                     has_subscribed=redditor.has_subscribed,
                                     hide_from_robots=redditor.hide_from_robots)
        else:  # No tine atributo 'id'
            print(
                f'redditor no tiene id: {redditor}')  # saber que objeto es (puesto que sino tiene id no debe ser un objeto de tipo redditor)
            if redditor.is_suspended:
                None
                # guardar los datos que traiga en este caso...
            elif redditor.is_banned:
                None
                # lo mismo...
            print(f'Redditor is_suspended: {redditor.is_suspended}')
            print(vars(redditor))
            db_author = None
    else:
        print('*********** NO insertado el \"Redditor\" actual. Se ha recibido "None" ***********')
        db_author = None
    return db_author


def saveSubmission(submission, redditor_nulo):
    query = select(Submission.id_submission).where(Submission.id_submission == submission.id)
    db_submission = db.session.execute(query).fetchone()  # devuelve uno

    if db_submission is not None:  # ya tenemos el submission
        # Sustituir por logger
        print("*********** NO insertado el \"Submission\" con id: " + submission.id + " *********** (Ya en la BD)")
        db_submission = None
    else:  # no tenemos este submission -> lo anhadimos
        if redditor_nulo:
            fk_id_author = 1
            id_author = 'null'
        else:
            query = select(Redditor.id).where(Redditor.id_redditor == submission.author.id)
            fk_id_author = db.session.execute(query).fetchone()[0]
            query = select(Redditor.id_redditor).where(Redditor.id_redditor == submission.author.id)
            id_author = db.session.execute(query).fetchone()[0]

        db_submission = Submission(id_submission=submission.id, title=submission.title,
                                   selftext=submission.selftext,
                                   fk_id_author=fk_id_author, id_author=id_author, ups=submission.ups,
                                   downs=submission.downs, upvote_ratio=submission.upvote_ratio, url=submission.url,
                                   link_flair_text=submission.link_flair_text.trim())  # REVISAR CORRECTO FUNCIONAMIENTO DE TRIM, PARA EVITAR DOS VALORES DE "PATIENT "
        return db_submission


def saveComment(comment, submission, redditor_nulo=False):
    db_comment = select(Comment.id).where(Comment.id_comment == comment.id)
    resultado = db.session.execute(db_comment).fetchone()  # devuelve uno

    if resultado is not None:  # ya tenemos el comment
        # Sustituir por logger
        print("*********** NO insertado el \"Comment\" con id: " + comment.id + " *********** (Ya en la BD)")
        db_comment = None
    else:  # no tenemos este comment -> lo anhadimos
        query = select(Submission.id).where(Submission.id_submission == submission.id)
        fk_id_submission = db.session.execute(query).fetchone()[0]

        if redditor_nulo:
            fk_id_author = 1
            id_author = 'null'
        else:
            query = select(Redditor.id).where(Redditor.id_redditor == comment.author.id)
            fk_id_author = db.session.execute(query).fetchone()[0]
            query = select(Redditor.id_redditor).where(Redditor.id_redditor == comment.author.id)
            id_author = db.session.execute(query).fetchone()[0]
        '''
            "parent_id" - The ID of the parent comment (prefixed with "t1_").
            If it is a top-level comment, this returns the submission ID instead
            (prefixed with "t3_").
        '''
        db_comment = Comment(id_comment=comment.id, fk_id_submission=fk_id_submission, id_submission=submission.id,
                             fk_id_author=fk_id_author, id_author=id_author, id_parent=comment.parent_id,
                             body=comment.body, ups=comment.ups, downs=comment.downs, depth=comment.depth)
    return db_comment

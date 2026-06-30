from models import User


class UserService:

    def __init__(self, db_session):
        self.db = db_session


    def create_user(self, email):
        user = User(email=email)

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user

    # --------------------------------
    # Get by ID
    # --------------------------------

    def get_user(self,user_id: str) -> User | None:
        return (
            self.db.query(User)
            .filter(
                User.id == user_id
            )
            .first()
        )

    # --------------------------------
    # Get by Email
    # --------------------------------

    def get_user_by_email(self,email: str) -> User | None:
        return (
            self.db.query(User)
            .filter(
                User.email == email
            )
            .first()
        )

    # --------------------------------
    # Exists
    # --------------------------------

    def user_exists(self,user_id: str) -> bool:
        return (
                self.db.query(User)
                .filter(
                    User.id == user_id
                )
                .first()
                is not None
        )

    # --------------------------------
    # Exists by Email
    # --------------------------------

    def email_exists(self,email: str) -> bool:
        return (
                self.db.query(User)
                .filter(
                    User.email == email
                )
                .first()
                is not None
        )

    # --------------------------------
    # Get or Create
    # --------------------------------

    def get_or_create_user(self,email: str) -> User:
        user = self.get_user_by_email(
            email
        )

        if user:
            return user

        return self.create_user(
            email
        )

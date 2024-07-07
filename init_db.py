from werkzeug.security import generate_password_hash
from app import app, db, User

with app.app_context():
    # Create all tables
    db.create_all()

    # Check if admin user already exists
    admin_user = User.query.filter_by(username='admin').first()
    if admin_user is None:
        # Create the admin user
        admin_user = User(username='admin', password=generate_password_hash('admin'), is_admin=True)
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created successfully.")
    else:
        print("Admin user already exists.")

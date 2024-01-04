---
layout: post
title: "(P1) flask tutorial"
date: 2023-08-31 20:00:04 +0800
labels: [web, flask]
---


```python
from flask import Flask
from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

bp = Blueprint("auth", __name__, url_prefix="/auth")

@bp.route("/register", methods=("GET", "POST"))
def register():
    # 重定向到 auth.py:login 函数对应的URL上, 
    url = url_for("auth.login")  # '/auth/login'
    redirect(url)  

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",
        # store the database in the instance folder
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.update(test_config)
    
    @app.route("/hello")
    def hello():
        return "Hello, World!"

    # register the database commands
    # from flaskr import db
    # db.init_app(app)

    # apply the blueprints to the app
    from flaskr import auth, blog

    app.register_blueprint(auth.bp)
    app.register_blueprint(blog.bp)

    # make url_for('index') == url_for('blog.index')
    # in another app, you might define a separate main index here with
    # app.route, while giving the blog blueprint a url_prefix, but for
    # the tutorial the blog will be the main index
    app.add_url_rule("/", endpoint="index")


    app.teardown_appcontext(close_db)

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()
```


```html
<form method="post">
    <label for="username">Username</label>  <!-- for 属性用于匹配 input 标签的 id 属性 -->
    <input name="username" id="username" required>  <!-- name 属性与 flask 代码中的 request.form["username"] 相对应 -->
    <label for="password">Password</label>
    <input type="password" name="password" id="password" required> <!-- type 是 input 标签的特有属性 -->
    <input type="submit" value="Register">
</form>
```
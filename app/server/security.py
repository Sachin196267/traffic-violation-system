"""Passwords, sessions, self-hosted CAPTCHA, verification codes, and a small rate limiter."""
import base64
import hashlib
import hmac
import io
import random
import secrets
import smtplib
import string
import time
from collections import defaultdict, deque
from email.message import EmailMessage

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config
import db

SECRET = (config.DATA / ".secret")
if not SECRET.exists():
    SECRET.write_bytes(secrets.token_bytes(32))
KEY = SECRET.read_bytes()


# ---------------- passwords ----------------
def hash_pw(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 310_000)
    return "pbkdf2$310000$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(h).decode()


def check_pw(pw: str, stored: str) -> bool:
    try:
        _, it, s, h = stored.split("$")
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(), base64.b64decode(s), int(it))
        return hmac.compare_digest(got, base64.b64decode(h))
    except Exception:
        return False


def password_problems(pw: str) -> list[str]:
    p = []
    if len(pw) < 10:
        p.append("at least 10 characters")
    if not any(c.islower() for c in pw) or not any(c.isupper() for c in pw):
        p.append("upper- and lower-case letters")
    if not any(c.isdigit() for c in pw):
        p.append("a number")
    if not any(c in string.punctuation for c in pw):
        p.append("a symbol")
    return p


# ---------------- sessions (opaque token in an HttpOnly cookie; only its hash is stored) ----------------
def _th(tok):
    return hmac.new(KEY, tok.encode(), hashlib.sha256).hexdigest()


def create_session(user_id: str) -> str:
    tok = secrets.token_urlsafe(32)
    with db.conn() as c:
        c.execute("INSERT INTO sessions VALUES (?,?,?,?)", (_th(tok), user_id, time.time(), time.time() + config.SESSION_DAYS * 86400))
    return tok


def session_user(tok: str | None):
    if not tok:
        return None
    with db.conn() as c:
        r = c.execute("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>?",
                      (_th(tok), time.time())).fetchone()
    return db.row(r)


def end_session(tok: str | None):
    if tok:
        with db.conn() as c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (_th(tok),))


# ---------------- CAPTCHA: distorted text image; answer bound to a signed, expiring token ----------------
ALPH = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def _sign(payload: str) -> str:
    return payload + "." + hmac.new(KEY, payload.encode(), hashlib.sha256).hexdigest()[:32]


def new_captcha():
    text = "".join(random.choice(ALPH) for _ in range(5))
    img = Image.new("RGB", (180, 60), (246, 247, 249))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    for _ in range(6):
        d.line([(random.randint(0, 180), random.randint(0, 60)) for _ in range(2)], fill=(170, 176, 186), width=1)
    x = 12
    for ch in text:
        ci = Image.new("RGBA", (40, 50), (0, 0, 0, 0))
        ImageDraw.Draw(ci).text((6, 4), ch, font=font, fill=(40, 48, 60))
        ci = ci.rotate(random.uniform(-22, 22), resample=Image.BICUBIC)
        img.paste(ci, (x, random.randint(0, 10)), ci)
        x += 32
    img = img.filter(ImageFilter.SMOOTH)
    buf = io.BytesIO(); img.save(buf, "PNG")
    exp = int(time.time()) + 300
    ans = hmac.new(KEY, (text + str(exp)).encode(), hashlib.sha256).hexdigest()[:24]
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), _sign(f"{exp}:{ans}")


_used_captchas: dict[str, float] = {}


def check_captcha(token: str, answer: str) -> bool:
    try:
        payload, sig = token.rsplit(".", 1)
        if _sign(payload) != token:
            return False
        exp, ans = payload.split(":")
        if int(exp) < time.time() or token in _used_captchas:
            return False
        ok = hmac.compare_digest(ans, hmac.new(KEY, (answer.strip().upper() + exp).encode(), hashlib.sha256).hexdigest()[:24])
        if ok:
            _used_captchas[token] = time.time()
        return ok
    except Exception:
        return False


# ---------------- verification codes ----------------
def send_code(target: str, purpose: str) -> dict:
    code = f"{secrets.randbelow(1_000_000):06d}"
    with db.conn() as c:
        c.execute("UPDATE codes SET used=1 WHERE target=? AND purpose=?", (target, purpose))
        c.execute("INSERT INTO codes VALUES (?,?,?,?,?,0,0)",
                  (db.new_id(), target, purpose, _th(code), time.time() + 600))
    subject = {"verify": "Verify your account", "reset": "Reset your password"}[purpose]
    body = f"Your verification code is {code}. It expires in 10 minutes."
    if config.DEV_MODE or "@" not in target:
        # no SMTP / SMS provider configured: developer outbox (clearly flagged in the UI)
        with open(config.DATA / "dev_outbox.log", "a") as f:
            f.write(f"{time.strftime('%F %T')}  to={target}  {subject}: {code}\n")
        return {"delivered": "dev_outbox", "dev_code": code}
    msg = EmailMessage(); msg["From"] = config.SMTP_FROM; msg["To"] = target; msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(config.SMTP_HOST, 465) as s:
        s.login(config.SMTP_USER, config.SMTP_PASS); s.send_message(msg)
    return {"delivered": "email"}


def check_code(target: str, purpose: str, code: str) -> bool:
    with db.conn() as c:
        r = c.execute("SELECT * FROM codes WHERE target=? AND purpose=? AND used=0 AND expires>? ORDER BY expires DESC LIMIT 1",
                      (target, purpose, time.time())).fetchone()
        if not r or r["attempts"] >= 5:
            return False
        ok = hmac.compare_digest(r["code_hash"], _th(code.strip()))
        c.execute("UPDATE codes SET attempts=attempts+1, used=? WHERE id=?", (1 if ok else 0, r["id"]))
    return ok


# ---------------- rate limit (per key, sliding window) ----------------
_hits: dict[str, deque] = defaultdict(deque)


def limited(key: str, n: int, window_s: float) -> bool:
    q = _hits[key]; t = time.time()
    while q and q[0] < t - window_s:
        q.popleft()
    if len(q) >= n:
        return True
    q.append(t)
    return False

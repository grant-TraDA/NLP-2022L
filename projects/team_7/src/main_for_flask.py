import os
from flask import Flask, render_template
import forms
import main_for_eden
import logging

logger = logging.getLogger("Poem-Generation")
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'askrhvbyuebiwndwl'
app.model, app.gpt2, app.t_tokenizer, app.a_tokenizer = main_for_eden.load_models()
logger.info("Initialisation finished")


@app.route("/", methods=["GET", "POST"])
def index():
    form = forms.MainForm()
    next_line = ""
    if form.validate_on_submit():
        if form.model_type.data == 'gpt2':
            next_line = main_for_eden.test_gpt2_on_line(
                app.gpt2, app.t_tokenizer, temp=float(form.temperature.data), zero_line_text=form.input.data)
        else:
            next_line = main_for_eden.test_model_and_gpt2_on_line(
                app.model, app.gpt2, app.t_tokenizer, app.a_tokenizer,
                temp=float(form.temperature.data), zero_line_text=form.input.data)

    return render_template("main.html", next_line=next_line, form=form)

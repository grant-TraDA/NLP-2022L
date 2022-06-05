from email.policy import default
from wtforms import TextAreaField, SubmitField, DecimalField, RadioField
from flask_wtf import FlaskForm


class MainForm(FlaskForm):
    model_type = RadioField(
        "Model type",
        choices=[("gpt2", "Only GPT2 model"),
                 ("both", "GPT2 with accent transformer")],
        default="gpt2")
    input = TextAreaField(
        "First line of poem", default="this definitely racks the joints this fires the veins")
    temperature = DecimalField("Temperature", default=0.8)
    submit = SubmitField("Generate >>")

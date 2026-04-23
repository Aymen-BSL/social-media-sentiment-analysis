from __future__ import annotations

from html import escape

from bootstrap import ensure_local_packages

ensure_local_packages()

from predict_ml import predict_comments
from predict_pretrained import predict_comments_with_vader


def _reaction_badge(reaction: str) -> str:
    colors = {
        "Liked": "#dff6dd",
        "Not Liked": "#ffe0e0",
        "Mixed Reaction": "#fff2c7",
    }
    return (
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;"
        f"background:{colors.get(reaction, '#eee')};font-weight:600'>{escape(reaction)}</span>"
    )


def _result_card(title: str, comments: list[str], labels: list[str], reaction: dict, scores=None) -> str:
    rows = []
    for index, (comment, label) in enumerate(zip(comments, labels), start=1):
        extra = ""
        if scores is not None:
            extra = f" <span style='color:#666'>(compound={scores[index - 1]['compound']:.3f})</span>"
        rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #eee'>{index}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #eee'>{escape(comment)}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #eee'><strong>{escape(label)}</strong>{extra}</td>"
            "</tr>"
        )

    counts = reaction["counts"]
    return f"""
    <div style="border:1px solid #ddd;border-radius:14px;padding:16px;margin-top:14px;background:#fff">
      <h4 style="margin:0 0 10px 0">{escape(title)}</h4>
      <div style="margin-bottom:10px">{_reaction_badge(reaction["reaction"])}</div>
      <div style="margin-bottom:12px;color:#444">
        positive={counts["positive"]} | negative={counts["negative"]} | neutral={counts["neutral"]} | total={counts["total"]}
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead>
          <tr style="background:#f7f7f7;text-align:left">
            <th style="padding:8px">#</th>
            <th style="padding:8px">Comment</th>
            <th style="padding:8px">Predicted sentiment</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def launch_live_demo() -> None:
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ImportError as exc:
        raise SystemExit(
            "Missing notebook demo dependency. Install it with: pip install -r requirements.txt"
        ) from exc

    post_title = widgets.Text(
        value="Campus Tech Club Launch",
        description="Title:",
        layout=widgets.Layout(width="100%"),
    )
    post_body = widgets.Textarea(
        value="We just launched a new student platform for club events, announcements, and project collaboration.",
        description="Post:",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    comments_box = widgets.Textarea(
        value="Love this idea!\nThis could be really useful.\nNot sure the design is clear yet.",
        description="Comments:",
        layout=widgets.Layout(width="100%", height="140px"),
    )
    analyze_button = widgets.Button(
        description="Analyze Live Comments",
        button_style="primary",
        icon="play",
    )
    output = widgets.Output()

    def on_click(_button) -> None:
        comments = [line.strip() for line in comments_box.value.splitlines() if line.strip()]
        with output:
            clear_output()
            if not comments:
                display(HTML("<b>Please enter at least one comment.</b>"))
                return

            display(
                HTML(
                    f"""
                    <div style="border:1px solid #ddd;border-radius:16px;padding:18px;background:#fafafa">
                      <div style="font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.08em">Mock Post</div>
                      <h3 style="margin:8px 0 10px 0">{escape(post_title.value)}</h3>
                      <p style="margin:0;color:#333;line-height:1.5">{escape(post_body.value)}</p>
                    </div>
                    """
                )
            )

            ml_result = predict_comments(comments)
            vader_result = predict_comments_with_vader(comments)

            display(HTML(_result_card("Trained ML Model", comments, ml_result["labels"], ml_result["reaction"])))
            display(
                HTML(
                    _result_card(
                        "VADER Baseline",
                        comments,
                        vader_result["labels"],
                        vader_result["reaction"],
                        scores=vader_result["scores"],
                    )
                )
            )

    analyze_button.on_click(on_click)

    display(
        widgets.VBox(
            [
                widgets.HTML(
                    "<h3 style='margin:0'>Live System Demo</h3>"
                    "<p style='margin:6px 0 12px 0'>Edit the mock post and type one comment per line, then click the button to simulate the app.</p>"
                ),
                post_title,
                post_body,
                comments_box,
                analyze_button,
                output,
            ]
        )
    )

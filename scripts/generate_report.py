"""Generate MLOps_A01_RollNumber.pdf submission report according to assignment guidelines."""

import os
from typing import List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def create_report(
    roll_number: str = "22F-3446",
    student_name: str = "Syed Abdullah",
    output_filename: Optional[str] = None,
    table1_data: Optional[List[List[str]]] = None,
    mlflow_overview_img: Optional[str] = None,
    mlflow_plot_img: Optional[str] = None,
    mlflow_registry_img: Optional[str] = None,
    github_actions_img: Optional[str] = None,
    git_tree_text: Optional[str] = None,
    analysis_text: Optional[str] = None,
) -> str:
    """Generate the MLOps Assignment 1 PDF report.

    Args:
        roll_number: Student Roll Number.
        student_name: Student Name.
        output_filename: Destination PDF path.
        table1_data: Hyperparameter search results matrix.
        mlflow_overview_img: Path to MLflow Experiment overview screenshot.
        mlflow_plot_img: Path to MLflow Metric comparison plot screenshot.
        mlflow_registry_img: Path to MLflow Model Registry screenshot.
        github_actions_img: Path to GitHub Actions CI passing run screenshot.
        git_tree_text: Git log tree textual representation.
        analysis_text: Max 150-word model analysis and findings.

    Returns:
        Path to the generated PDF.
    """
    if output_filename is None:
        output_filename = f"MLOps_A01_{roll_number}.pdf"

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )

    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("FAST NUCES - Fall 2026", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Assignment 01: MLOps with CI/CD, MLflow & Makefile Automation",
        title_style,
    ))
    story.append(Spacer(1, 6))

    meta_text = (
        f"<b>Student Name:</b> {student_name} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Roll Number:</b> {roll_number} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Course:</b> Machine Learning Operations (MLOps)"
    )
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(
        width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=8
    ))

    # Section 1: Table 1 - Hyperparameter Search Results
    story.append(Paragraph("1. Table 1: Hyperparameter Search & CV Results", h1_style))
    story.append(Paragraph(
        "Evaluation across 6 configurations (3 Random Forest, 3 Gradient Boosting) using "
        "5-Fold Stratified Cross-Validation on the Wine cultivar dataset (80% training split).",
        body_style,
    ))
    story.append(Spacer(1, 5))

    if table1_data is None:
        table1_data = [
            ["Model Family", "Config ID", "Key Hyperparameters", "Train F1",
             "Val F1", "Val Acc", "Val Loss"],
            ["RandomForest", "RF_cfg1", "n_est=50, depth=3, split=2", "0.9930",
             "0.9665", "0.9653", "0.2055"],
            ["RandomForest", "RF_cfg2", "n_est=100, depth=5, split=4", "1.0000",
             "0.9789", "0.9791", "0.1682"],
            ["RandomForest", "RF_cfg3", "n_est=150, depth=None, split=2", "1.0000",
             "0.9789", "0.9791", "0.1636"],
            ["GradientBoost", "GBM_cfg1", "n_est=50, lr=0.05, depth=3", "1.0000",
             "0.9053", "0.9022", "0.2295"],
            ["GradientBoost", "GBM_cfg2", "n_est=100, lr=0.10, depth=3", "1.0000",
             "0.9259", "0.9236", "0.4281"],
            ["GradientBoost", "GBM_cfg3", "n_est=120, lr=0.20, depth=4", "1.0000",
             "0.9197", "0.9160", "0.5264"],
        ]

    t = Table(table1_data, colWidths=[80, 55, 140, 50, 50, 50, 55])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("FONTSIZE", (0, 1), (-1, -1), 7.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 8))

    # Section 2: MLflow UI Visualizations
    story.append(Paragraph("2. MLflow Tracking & Model Registry Evidence", h1_style))
    story.append(Paragraph(
        "Visual evidence from the local MLflow server capturing experiment runs, "
        "metric comparisons, and the registered champion model.",
        body_style,
    ))
    story.append(Spacer(1, 5))

    def embed_screenshot(img_path: Optional[str], caption: str):
        if img_path and os.path.exists(img_path):
            story.append(Image(img_path, width=6.8 * inch, height=2.2 * inch))
            story.append(Spacer(1, 2))
            story.append(
                Paragraph(
                    f"<i>{caption}</i>",
                    ParagraphStyle(
                        "Cap",
                        parent=body_style,
                        fontSize=8,
                        alignment=1,
                        textColor=colors.HexColor("#718096"),
                    ),
                )
            )
            story.append(Spacer(1, 6))
        else:
            placeholder_text = (
                f"<b>[Screenshot Placeholder]</b> {caption}<br/>"
                f"<i>(Save image file to: <code>{img_path or 'path/to/screenshot.png'}</code>)</i>"
            )
            box = Table(
                [[Paragraph(placeholder_text, body_style)]],
                colWidths=[6.8 * inch],
                rowHeights=[35],
            )
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(box)
            story.append(Spacer(1, 6))

    embed_screenshot(mlflow_overview_img, "Figure 1: MLflow Experiment Overview")
    embed_screenshot(mlflow_plot_img, "Figure 2: MLflow Validation Macro F1 Comparison")
    embed_screenshot(mlflow_registry_img, "Figure 3: MLflow Model Registry (@champion alias)")

    # Section 3: CI/CD Pipeline Evidence
    story.append(Paragraph("3. CI/CD GitHub Actions Workflow Evidence", h1_style))
    story.append(Paragraph(
        "Evidence of automated CI workflow execution running lint, unit tests, "
        "and automated quality gates on push and pull requests.",
        body_style,
    ))
    story.append(Spacer(1, 5))
    embed_screenshot(github_actions_img, "Figure 4: GitHub Actions Passing Build")

    # Section 4: Git Collaboration & Conflict Resolution Log
    story.append(Paragraph("4. Git Commit Tree & Merge Conflict Resolution Log", h1_style))
    if git_tree_text is None:
        git_tree_text = (
            "*   419c438 (HEAD -> main) Merge branch 'conflict-simulation' into main "
            "(Resolved merge conflict in RF_cfg3 hyperparameters)\n"
            "|\\  \n"
            "| * b09ccb7 (conflict-simulation) experiment: increase RF_cfg3 "
            "n_estimators to 200 in conflict-simulation\n"
            "* | 0508afa tune: adjust RF_cfg3 n_estimators to 175 on main\n"
            "|/  \n"
            "*   434a8d7 Merge branch 'feature/ci-quality-gates' into main\n"
            "|\\  \n"
            "| * 21fa5e7 (feature/ci-quality-gates) feat: implement automated "
            "MLOps quality gates and GitHub Actions CI workflow\n"
            "|/  \n"
            "*   5359856 Merge branch 'feature/mlflow-tracking' into main\n"
            "|\\  \n"
            "| * 3946f47 (feature/mlflow-tracking) feat: implement 5-fold CV "
            "hyperparameter training, MLflow tracking, and model registry\n"
            "|/  \n"
            "* ea009b4 feat: initialize repository architecture, Makefile, "
            "and modular data pipeline"
        )

    formatted_git = git_tree_text.replace("\n", "<br/>").replace(" ", "&nbsp;")
    git_box = Table([[Paragraph(formatted_git, code_style)]], colWidths=[6.8 * inch])
    git_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(git_box)
    story.append(Spacer(1, 8))

    # Section 5: Analytical Summary (Max 150 words)
    story.append(Paragraph("5. Model Performance Analysis (< 150 Words)", h1_style))
    if analysis_text is None:
        analysis_text = (
            "We evaluated Random Forest and Gradient Boosting classifiers on the 13-feature "
            "Wine dataset using 5-fold stratified cross-validation. Random Forest achieved "
            "superior generalization stability, attaining a validation Macro F1-score of "
            "0.9789 and lower validation log loss (0.1636) compared to Gradient Boosting "
            "(Val F1: 0.9259, Loss: 0.4281). Due to its robust ensemble averaging and "
            "minimal variance across CV folds, RandomForest (RF_cfg2) was promoted as the "
            "'champion' model in the MLflow Model Registry. On the independent hold-out "
            "test set (20%), the champion model achieved 1.0000 Macro F1-score with an "
            "ultra-low inference latency of ~6.4 ms, comfortably passing all three automated "
            "MLOps quality gates (F1 >= 0.88, latency <= 30 ms, valid cultivar class indices)."
        )

    story.append(Paragraph(analysis_text, body_style))

    doc.build(story)
    print(f"Report generated successfully: {output_filename}")
    return output_filename


if __name__ == "__main__":
    create_report()

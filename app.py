import gradio as gr
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Prediction wrapper
def predict_job_success(
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    Work_accident,
    promotion_last_5years,
    Department,
    salary,
):
    try:
        # Build custom data object
        data = CustomData(
            satisfaction_level=float(satisfaction_level),
            last_evaluation=float(last_evaluation),
            number_project=int(number_project),
            average_montly_hours=int(average_montly_hours),
            time_spend_company=int(time_spend_company),
            Work_accident=int(Work_accident),
            promotion_last_5years=int(promotion_last_5years),
            Department=Department,
            salary=salary
        )

        df = data.get_data_as_data_frame()

        # Run prediction pipeline
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)[0]

        return "Will Stay in Job" if prediction == 0 else "Will Leave the Job"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Gradio UI
iface = gr.Interface(
    fn=predict_job_success,
    inputs=[
        gr.Slider(0.0, 1.0, step=0.01, label="Satisfaction Level"),
        gr.Slider(0.0, 1.0, step=0.01, label="Last Evaluation"),
        gr.Slider(1, 10, step=1, label="Number of Projects"),
        gr.Slider(80, 320, step=1, label="Average Monthly Hours"),
        gr.Slider(1, 10, step=1, label="Time Spent at Company"),
        gr.Radio([0, 1], label="Work Accident (0 = No, 1 = Yes)"),
        gr.Radio([0, 1], label="Promotion in Last 5 Years (0 = No, 1 = Yes)"),
        gr.Dropdown(["sales", "technical", "support", "management", "IT", "HR", "product_mng", "marketing", "RandD", "accounting"], label="Department"),
        gr.Dropdown(["low", "medium", "high"], label="Salary Level")
    ],
    outputs="text",
    title="Job Success Prediction"
)

if __name__ == "__main__":
    iface.launch()

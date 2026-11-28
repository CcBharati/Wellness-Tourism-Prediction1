# Wellness-Tourism-Prediction1 
- name: Upload Model and Evaluation Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: wellness-model-and-artifacts
          path: |
            best_wellness_model.joblib
            best_model.pkl
            models/
            *.csv
            *.png
          if-no-files-found: warn

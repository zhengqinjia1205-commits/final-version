import re

with open('forecastpro.py', 'r', encoding='utf-8') as f:
    content = f.read()

export_code = """
    def export_combined_excel(self, periods: int = 4, output_dir: str = "./reports"):
        import os
        import pandas as pd
        from datetime import datetime

        if self.data is None or self.train_data is None or self.test_data is None:
            return

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"forecast_report_{ts}.xlsx")

        y_full = self.data[self.demand_col].astype(float)
        train_idx = self.train_data["y"].index
        test_idx = self.test_data["y"].index

        last_date = self.data.index[-1]
        try:
            future_idx = pd.date_range(start=last_date, periods=int(periods) + 1, freq=self.freq)[1:]
        except Exception:
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=int(periods), freq="D")

        all_idx = train_idx.union(test_idx).union(future_idx)
        df_out = pd.DataFrame(index=all_idx)
        df_out["Actual"] = y_full

        df_out["Set"] = ""
        df_out.loc[train_idx, "Set"] = "Train"
        df_out.loc[test_idx, "Set"] = "Test"
        df_out.loc[future_idx, "Set"] = "Future"

        best_model = getattr(self, "best_model", None)
        if best_model:
            fitted = None
            if (self.baseline_models or {}).get(best_model) is not None:
                fitted = self.baseline_models[best_model].get("fitted")
            elif (self.advanced_models or {}).get(best_model) is not None:
                fitted = self.advanced_models[best_model].get("fitted")
            
            if fitted is not None:
                n = min(len(train_idx), len(fitted))
                df_out.loc[train_idx[:n], f"Fitted ({best_model})"] = fitted[:n]

            preds = None
            if (self.baseline_models or {}).get(best_model) is not None:
                preds = self.baseline_models[best_model].get("predictions")
            elif (self.advanced_models or {}).get(best_model) is not None:
                preds = self.advanced_models[best_model].get("predictions")
            
            if preds is not None:
                n = min(len(test_idx), len(preds))
                df_out.loc[test_idx[:n], f"Test_Predicted ({best_model})"] = preds[:n]
        
        fc_res = getattr(self, "forecast_results", None)
        if fc_res and fc_res.get("forecast"):
            fc = fc_res["forecast"]
            n = min(len(future_idx), len(fc))
            df_out.loc[future_idx[:n], f"Future_Forecast ({fc_res.get('model', 'auto')})"] = fc[:n]
            if fc_res.get("lower_bound"):
                df_out.loc[future_idx[:n], "Future_Lower_Bound"] = fc_res["lower_bound"][:n]
            if fc_res.get("upper_bound"):
                df_out.loc[future_idx[:n], "Future_Upper_Bound"] = fc_res["upper_bound"][:n]

        try:
            df_out.reset_index(names=[self.time_col]).to_excel(filepath, index=False)
            print(f"Combined excel report generated at {filepath}")
        except Exception as e:
            print(f"Failed to generate combined excel report: {e}")
"""

if "def export_combined_excel" not in content:
    # Add it to the ForecastProAgent class
    content = content.replace("    def export_method_folders(self", export_code + "\n    def export_method_folders(self")
    with open('forecastpro.py', 'w', encoding='utf-8') as f:
        f.write(content)

with open('backend/main.py', 'r', encoding='utf-8') as f:
    main_content = f.read()

if "agent.export_combined_excel" not in main_content:
    main_content = main_content.replace(
        "agent.export_method_folders(periods=max(4, int(periods)))",
        "agent.export_method_folders(periods=max(4, int(periods)))\n            agent.export_combined_excel(periods=max(4, int(periods)))"
    )
    
    # Also add to agent_predict route
    main_content = main_content.replace(
        """                except Exception as e:
                    future_errors[m] = str(e)

            return {""",
        """                except Exception as e:
                    future_errors[m] = str(e)
            
            try:
                agent.export_method_folders(periods=max(4, int(periods)))
                agent.export_combined_excel(periods=max(4, int(periods)))
            except Exception as e:
                print(f"Method export error in agent route: {e}")

            return {"""
    )
    
    with open('backend/main.py', 'w', encoding='utf-8') as f:
        f.write(main_content)

print("Export logic added.")

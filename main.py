import pandas as pd
import numpy as np
import yfinance as yf
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt

def main():
    print("Hello from nifty-rbi-causal!")


if __name__ == "__main__":
    main()

    # 1. Download Nifty50 data
    nifty = yf.download('^NSEI', start='2010-01-01', end='2025-07-20').reset_index()

    print(nifty.columns)

    nifty.columns = [
        '_'.join(filter(None, map(str, col))) if isinstance(col, tuple) else col
        for col in nifty.columns
    ]

    # print(nifty.columns)

    # 2. Sample RBI events (replace with your own, ideally append full event history)
    rbi_events = pd.DataFrame({
        'Date': ['2023-02-08', '2022-12-07', '2022-09-30', '2022-08-05', '2022-06-08', '2020-05-22'],
        'Rate': [6.5, 6.25, 5.9, 5.4, 4.9, 4.0]
    })
    rbi_events['Date'] = pd.to_datetime(rbi_events['Date'])
    rbi_events['Delta'] = rbi_events['Rate'].diff().fillna(0)
    rbi_events['Event'] = np.where(rbi_events['Delta'] < 0, 'cut',
                                np.where(rbi_events['Delta'] > 0, 'hike', 'no_change'))

    # 3. Merge with stock data (label each day with latest RBI event)
    nifty['Date'] = pd.to_datetime(nifty['Date'])
    print(nifty.sort_values('Date').shape)
    print("test")
    print(rbi_events[['Date', 'Event']].shape)
    df = pd.merge_asof(nifty.sort_values('Date'),
                    rbi_events[['Date', 'Event']].sort_values('Date'),
                    on='Date',
                    direction='backward')
    df['Event'].fillna('no_event', inplace=True)
    df['Return'] = df['Close_^NSEI'].pct_change().shift(-1)   # next day's return

    print(df)
    # 4. Define treatment: did a rate cut happen
    df['Treatment'] = (df['Event'] == 'cut').astype(int)
    # Drop rows with NA in outcome/treatment/covariates
    df = df.dropna(subset=['Return', 'Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Volume_^NSEI', 'Treatment'])

    # Minimal covariates for illustration
    covariates = ['Open_^NSEI', 'High_^NSEI', 'Low_^NSEI', 'Volume_^NSEI']

    # 5. Build causal model in DoWhy
    model = CausalModel(
        data = df,
        treatment = 'Treatment',
        outcome = 'Return',
        common_causes = covariates,
        instruments = None
    )
    model.view_model(layout="dot")  # Optional: shows a graph (requires graphviz installed)

    # 6. Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Identified estimand:", identified_estimand)

    # 7. Estimate using backdoor Linear Regression (or try other methods)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    print(f"\nCausal effect of rate cut on Nifty 50 next day return: {estimate.value:.6f}")

    # 8. (Optional) Refutation
    refute = model.refute_estimate(
        identified_estimand, estimate, method_name="placebo_treatment_refuter"
    )
    print("\nRefutation result:", refute)

    # 9. Plot returns by treatment for visual sanity check
    df.boxplot(column='Return', by='Treatment')
    plt.title('Nifty 50 Next-day Returns by Rate-Cut Event')
    plt.suptitle('')
    plt.xlabel('Rate Cut Event (0=No, 1=Yes)')
    plt.ylabel('Next-day Return')
    plt.show()

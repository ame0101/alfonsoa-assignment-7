from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    error = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, label="Data Points", alpha=0.5)
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Regression Line")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    sim_model = LinearRegression()  # Create model once
    X_reshaped = np.zeros((N, 1))  # Pre-allocate array

    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim
        
        np.copyto(X_reshaped, X_sim.reshape(-1, 1))  # Reuse array
        sim_model.fit(X_reshaped, Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color="skyblue", edgecolor="black")
    plt.title("Histogram of Slopes")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color="salmon", edgecolor="black")
    plt.title("Histogram of Intercepts")
    plt.xlabel("Intercept")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    return (X, Y, slope, intercept, plot1_path, plot2_path, 
            slope_more_extreme, intercept_extreme, slopes, intercepts)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme,
         slopes, intercepts) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template("index.html",
                             plot1=plot1,
                             plot2=plot2,
                             slope_extreme=slope_extreme,
                             intercept_extreme=intercept_extreme,
                             N=N, mu=mu, sigma2=sigma2,
                             beta0=beta0, beta1=beta1, S=S)
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= 
                         np.abs(observed_stat - hypothesized_value))
    else:
        p_value = None

    fun_message = "Wow! Extremely small p-value! This is a rare event." if p_value and p_value <= 0.0001 else None

    # Plot histogram
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=30, color="lightgreen", edgecolor="black")
    plt.axvline(x=observed_stat, color="red", linestyle="--", label="Observed Statistic")
    plt.axvline(x=hypothesized_value, color="blue", linestyle="--", label="Hypothesized Value")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s")
    plt.xlabel(f"{parameter.capitalize()} Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    return render_template("index.html",
                         plot1="static/plot1.png",
                         plot2="static/plot2.png",
                         plot3=plot3_path,
                         parameter=parameter,
                         observed_stat=observed_stat,
                         hypothesized_value=hypothesized_value,
                         N=N, beta0=beta0, beta1=beta1, S=S,
                         p_value=p_value,
                         fun_message=fun_message)

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use stored simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate confidence interval
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    
    alpha = 1 - (confidence_level / 100)
    t_crit = stats.t.ppf(1 - alpha/2, df=S-1)
    margin_of_error = t_crit * (std_estimate / np.sqrt(S))
    
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error
    
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot confidence interval
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(estimates)), estimates, color="gray", alpha=0.5, label="Estimates")
    plt.hlines(y=mean_estimate, xmin=0, xmax=S, colors="blue", linestyles="--", 
              label="Mean Estimate")
    plt.hlines(y=[ci_lower, ci_upper], xmin=0, xmax=S, colors="red", linestyles="--",
              label=f"{confidence_level}% Confidence Interval")
    plt.axhline(y=true_param, color="green", linestyle="-", linewidth=2,
                label="True Parameter")
    plt.xlabel("Simulation Number")
    plt.ylabel(f"{parameter.capitalize()} Estimate")
    plt.title(f"Confidence Interval for {parameter.capitalize()}")
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    return render_template("index.html",
                         plot1="static/plot1.png",
                         plot2="static/plot2.png",
                         plot4=plot4_path,
                         parameter=parameter,
                         confidence_level=confidence_level,
                         mean_estimate=mean_estimate,
                         ci_lower=ci_lower,
                         ci_upper=ci_upper,
                         includes_true=includes_true,
                         observed_stat=observed_stat,
                         N=N, mu=mu, sigma2=sigma2,
                         beta0=beta0, beta1=beta1, S=S)

if __name__ == "__main__":
    app.run(debug=True)

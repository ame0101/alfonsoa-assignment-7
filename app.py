from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N)
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error
    
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    plot1_path = "static/plot1.png"
    plt.figure(figsize=(5, 5))
    plt.scatter(X, Y, color='blue', alpha=0.5)
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Regression Line: Y = {slope:.2f}X + {intercept:.2f}')
    plt.tight_layout()
    plt.savefig(plot1_path)
    plt.close()
    
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim
        
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_
        
        slopes.append(sim_slope)
        intercepts.append(sim_intercept)
    
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Observed Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Observed Intercept: {intercept:.2f}")
    plt.axvline(beta1, color="green", linestyle="--", linewidth=1, label=f"True Slope: {beta1:.2f}")
    plt.axvline(beta0, color="red", linestyle="--", linewidth=1, label=f"True Intercept: {beta0:.2f}")
    plt.title("Histogram of Simulated Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()
    
    slope_more_extreme = np.sum(slopes >= slope) / S
    intercept_extreme = np.sum(intercepts <= intercept) / S
    
    return X, Y, slope, intercept, plot1_path, plot2_path, slope_more_extreme, intercept_extreme, slopes, intercepts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])
        
        X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme, slopes, intercepts = generate_data(N, mu, beta0, beta1, sigma2, S)
        
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
        
        return render_template("index.html", plot1=plot1, plot2=plot2, slope_extreme=slope_extreme, intercept_extreme=intercept_extreme,
                               N=N, mu=mu, sigma2=sigma2, beta0=beta0, beta1=beta1, S=S)
    
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
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
    
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0
    
    if test_type == 'greater':
        p_value = np.sum(simulated_stats >= observed_stat) / S
    elif test_type == 'less':
        p_value = np.sum(simulated_stats <= observed_stat) / S
    else:  # 'not_equal'
        p_value = np.sum(np.abs(simulated_stats - observed_stat) >= np.abs(observed_stat - hypothesized_value)) / S
    
    fun_message = None
    if p_value <= 0.0001:
        fun_message = "Wow! You've encountered an extremely rare event!"
    
    plot3_path = "static/plot3.png"
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, alpha=0.5, color="blue", label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", linewidth=1, label=f"Observed {parameter.capitalize()}: {observed_stat:.2f}")
    plt.axvline(hypothesized_value, color="green", linestyle="--", linewidth=1, label=f"Hypothesized Value: {hypothesized_value:.2f}")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s")
    plt.xlabel(f"{parameter.capitalize()}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot3_path)
    plt.close()
    
    return render_template("index.html", plot1="static/plot1.png", plot2="static/plot2.png", plot3=plot3_path, 
                           parameter=parameter, observed_stat=observed_stat, hypothesized_value=hypothesized_value,
                           N=N, beta0=beta0, beta1=beta1, S=S, p_value=p_value, fun_message=fun_message)

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))
    
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0
    
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)
    
    df = len(estimates) - 1
    t_crit = t.ppf((1 + confidence_level / 100) / 2, df)
    se = std_estimate / np.sqrt(len(estimates))
    ci_lower = mean_estimate - t_crit * se
    ci_upper = mean_estimate + t_crit * se
    
    includes_true = ci_lower <= true_param <= ci_upper
    
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(estimates)), estimates, color="gray", alpha=0.5, label="Individual Estimates")
    plt.axhline(mean_estimate, color="blue", linestyle="-", linewidth=1, label="Mean Estimate")
    plt.axhline(ci_lower, color="orange", linestyle="--", linewidth=1, label=f"{confidence_level}% CI Lower Bound")
    plt.axhline(ci_upper, color="orange", linestyle="--", linewidth=1, label=f"{confidence_level}% CI Upper Bound")
    plt.axhline(true_param, color="red" if includes_true else "gray", linestyle="-", linewidth=1, label="True Parameter Value")
    plt.title(f"Confidence Interval for {parameter.capitalize()}")
    plt.xlabel("Simulation Index")
    plt.ylabel(f"{parameter.capitalize()} Estimate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot4_path)
    plt.close()
    
    return render_template("index.html", plot1="static/plot1.png", plot2="static/plot2.png", plot4=plot4_path,
                           parameter=parameter, confidence_level=confidence_level, mean_estimate=mean_estimate,
                           ci_lower=ci_lower, ci_upper=ci_upper, includes_true=includes_true, observed_stat=observed_stat,
                           N=N, mu=mu, sigma2=sigma2, beta0=beta0, beta1=beta1, S=S)

if __name__ == "__main__":
    app.run(debug=True)
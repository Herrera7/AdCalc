import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from IPython.display import clear_output
import seaborn as sns
import random

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def plot_euclidean_connection(a,b):
	ax  = plt.gca()
	eud = mlines.Line2D([a[0],b[0]], [a[1],b[1]])
	l1  = mlines.Line2D([a[0],b[0]], [a[0],a[0]], color="grey")
	l2  = mlines.Line2D([b[0],b[0]], [a[0],b[1]], color="grey")
	ax.add_line(eud)
	ax.add_line(l1)
	ax.add_line(l2)

def plot_euclidean_distance(x,y,sel_a,sel_b):
	ax = plt.gca()
	l = mlines.Line2D([x[sel_a],x[sel_b]], [y[sel_a],y[sel_b]])
	ax.add_line(l)

def plot_euclidean_distances(sel_point,x,y):
	ax = plt.gca()
	other_points  = [i for i in range(x.shape[0]) if i !=sel_point]
	sel_a = sel_point
	for oth_point in other_points:
		sel_b = oth_point
		l = mlines.Line2D([x[sel_a],x[sel_b]], [y[sel_a],y[sel_b]])
		ax.add_line(l)

def plot_manhattan_distance(x,y,sel_a,sel_b):
    ax = plt.gca()
    l1 = mlines.Line2D([x[sel_a],x[sel_a]], [y[sel_a],y[sel_b]])
    l2 = mlines.Line2D([x[sel_a],x[sel_b]], [y[sel_a],y[sel_a]])

    ax.add_line(l1)
    ax.add_line(l2)

def plot_breaking_kmeans_assumptions(n_samples = 1500,random_state = 170):
	plt.figure(figsize=(12, 12))	
	
	X, y = make_blobs(n_samples=n_samples, random_state=random_state)

	# Incorrect number of clusters
	y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

	plt.subplot(221)
	plt.scatter(X[:, 0], X[:, 1], c=y_pred)
	plt.title("Incorrect Number of Blobs")

	# Anisotropicly distributed data
	transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
	X_aniso = np.dot(X, transformation)
	y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

	plt.subplot(222)
	plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
	plt.title("Anisotropicly Distributed Blobs")

	# Different variance
	X_varied, y_varied = make_blobs(n_samples=n_samples,
	                                cluster_std=[1.0, 2.5, 0.5],
	                                random_state=random_state)
	y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

	plt.subplot(223)
	plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
	plt.title("Unequal Variance")

	# Unevenly sized blobs
	X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
	y_pred = KMeans(n_clusters=3,
	                random_state=random_state).fit_predict(X_filtered)

	plt.subplot(224)
	plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
	plt.title("Unevenly Sized Blobs")

	plt.show()

def generate_cluster_data(n_clusters = 4,nsamp = 100,seed=6,plotit=True):
    np.random.seed(seed)
    clus_dt_list = []
    centroids = np.random.uniform(0,10,size=[n_clusters,2])
    plt.figure(figsize=[8,6])
    for c in range(n_clusters):
        x = np.random.normal(centroids[c][0],size=nsamp)
        y = np.random.normal(centroids[c][1],size=nsamp)
        clus = [c]*nsamp
        clus_dt_list.append(pd.DataFrame({"X":x,"Y":y,"Cluster":clus}))
        if plotit:
            plt.scatter(x,y)
    if plotit:
        plt.show()
        
    return pd.concat(clus_dt_list)

def plot_predicted_clusters(stylized_dt):
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    for c in stylized_dt.Cluster_pred.unique():
        cluster_mask = (stylized_dt.Cluster_pred==c)
        stylized_dt.loc[cluster_mask,["X","Y"]].plot(kind="scatter",x="X",y="Y", 
                                                  color=cmap(c),
                                                  grid=True,s=40,alpha=0.7,
                                                  figsize=[8,6], ax=ax)

def plot_actual_clusters(stylized_dt,actual_field = "CLIENT_TYPE"):
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    for c in stylized_dt[actual_field].unique():
        cluster_mask = (stylized_dt[actual_field]==c)
        stylized_dt.loc[cluster_mask,["X","Y"]].plot(kind="scatter",x="X",y="Y", 
                                                  color=cmap(c),
                                                  grid=True,s=40,alpha=0.7,
                                                  figsize=[8,6], ax=ax)

def plot_pca_example():
	angle = np.pi / 5
	stretch = 5
	m = 200

	np.random.seed(3)
	X = np.random.randn(m, 2) / 10
	X = X.dot(np.array([[stretch, 0],[0, 1]])) # stretch
	X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]) # rotate

	u1 = np.array([np.cos(angle), np.sin(angle)])
	u2 = np.array([np.cos(angle - 2 * np.pi/6), np.sin(angle - 2 * np.pi/6)])
	u3 = np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])

	X_proj1 = X.dot(u1.reshape(-1, 1))
	X_proj2 = X.dot(u2.reshape(-1, 1))
	X_proj3 = X.dot(u3.reshape(-1, 1))

	plt.figure(figsize=(8,4))
	plt.subplot2grid((3,2), (0, 0), rowspan=3)
	plt.plot([-1.4, 1.4], [-1.4*u1[1]/u1[0], 1.4*u1[1]/u1[0]], "k-", linewidth=1)
	plt.plot([-1.4, 1.4], [-1.4*u2[1]/u2[0], 1.4*u2[1]/u2[0]], "k--", linewidth=1)
	plt.plot([-1.4, 1.4], [-1.4*u3[1]/u3[0], 1.4*u3[1]/u3[0]], "k:", linewidth=2)
	plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
	plt.axis([-1.4, 1.4, -1.4, 1.4])
	plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
	plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k', ec='k')
	plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
	plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
	plt.xlabel("$x_1$", fontsize=18)
	plt.ylabel("$x_2$", fontsize=18, rotation=0)
	plt.grid(True)

	plt.subplot2grid((3,2), (0, 1))
	plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
	plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
	plt.gca().get_yaxis().set_ticks([])
	plt.gca().get_xaxis().set_ticklabels([])
	plt.axis([-2, 2, -1, 1])
	plt.grid(True)

	plt.subplot2grid((3,2), (1, 1))
	plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
	plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
	plt.gca().get_yaxis().set_ticks([])
	plt.gca().get_xaxis().set_ticklabels([])
	plt.axis([-2, 2, -1, 1])
	plt.grid(True)

	plt.subplot2grid((3,2), (2, 1))
	plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
	plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
	plt.gca().get_yaxis().set_ticks([])
	plt.axis([-2, 2, -1, 1])
	plt.xlabel("$z_1$", fontsize=18)
	plt.grid(True)

	plt.show()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def plot_yield_curve_evolution(pca_data,days=20):
    for i in range(days):
        pca_data.iloc[i,:].plot()
        plt.show()
        time.sleep(1)
        clear_output()

def plot_cov_heatmap(cov):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 7))
    # Generate a custom diverging colormap
    sns.set()
    sns.heatmap(cov,cmap="YlGnBu",square=True, linewidths=.5, cbar_kws={"shrink": .5})

def generate_knn_data(sample_size):
    # Parameters for mean distributions
    mean_blue = [1, 0]
    mean_orange = [0, 1]
    mean_cov = np.eye(2)
    mean_size = 10
    
    # Additional parameters for blue and orange distributions
    sample_cov = np.eye(2)/5
    
    # Generate mean components for blue and orange (10 means for each)
    sample_blue_mean = np.random.multivariate_normal(mean_blue, mean_cov, mean_size)
    sample_orange_mean = np.random.multivariate_normal(mean_orange, mean_cov, mean_size)
    
    # Generate blue points
    sample_blue = np.array([
        np.random.multivariate_normal(sample_blue_mean[random.randint(0, 9)],
                                      sample_cov)
        for _ in range(sample_size)
    ])
    y_blue = [0 for _ in range(sample_size)]

    # Generate orange points
    sample_orange = np.array([
        np.random.multivariate_normal(sample_orange_mean[random.randint(0, 9)],
                                      sample_cov)
        for _ in range(sample_size)
    ])
    y_orange = [1 for _ in range(sample_size)]

    data_x = np.concatenate((sample_blue, sample_orange), axis=0)
    data_y = np.concatenate((y_blue, y_orange))
    return data_x, data_y

def plot_sample(sample_blue,sample_orange):
	# Plot
	fig = plt.figure(figsize=(15, 15))
	ax1 = fig.add_subplot(2, 2, 1)

	ax1.plot(sample_blue[:, 0], sample_blue[:, 1], 'o')
	ax1.plot(sample_orange[:, 0], sample_orange[:, 1], 'o', color='orange')
	ax1.set_title('0/1 Response')
	plt.show()

	plot_x_min, plot_x_max = ax1.get_xlim()
	plot_y_min, plot_y_max = ax1.get_ylim()

	return plot_x_min, plot_x_max,plot_y_min, plot_y_max

def plot_linear_classification(ols_beta, sample_blue,sample_orange, plot_x_min, plot_x_max,plot_y_min, plot_y_max):

	# Plot for OLS
	fig = plt.figure(figsize=(15, 15))
	ax2 = fig.add_subplot(2, 2, 2)

	ax2.plot(sample_blue[:, 0], sample_blue[:, 1], 'o', color='C0')
	ax2.plot(sample_orange[:, 0], sample_orange[:, 1], 'o', color='orange')

	# OLS line for y_hat = 0.5
	ols_line_y_min = (.5 - ols_beta[0] - plot_x_min*ols_beta[1])/ols_beta[2]
	ols_line_y_max = (.5 - ols_beta[0] - plot_x_max*ols_beta[1])/ols_beta[2]
	ax2.plot([plot_x_min, plot_x_max], [ols_line_y_min, ols_line_y_max], color='black')

	# https://matplotlib.org/examples/pylab_examples/fill_between_demo.html
	ax2.fill_between([plot_x_min, plot_x_max], plot_y_min, [ols_line_y_min, ols_line_y_max],
	                facecolor='blue', alpha=.2)
	ax2.fill_between([plot_x_min, plot_x_max], [ols_line_y_min, ols_line_y_max], plot_y_max,
	                facecolor='orange', alpha=.2)
	ax2.set_title('Linear Regression of 0/1 Response')
	ax2.set_xlim((plot_x_min, plot_x_max))
	ax2.set_ylim((plot_y_min, plot_y_max))
	plt.plot()
	
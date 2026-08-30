import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.001)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  # Normalize pixel values to [0, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"mlp-lr-{args.lr}-alpha-{args.alpha}"):
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("alpha", args.alpha)

        model = MLPClassifier(
            hidden_layer_sizes = (100,), 
            learning_rate_init=args.lr, 
            alpha=args.alpha, 
            max_iter = 20,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("training loss", model.loss_)
        mlflow.sklearn.log_model(model, 
                                 name="model",
                                 skops_trusted_types = ['sklearn.neural_network._stochastic_optimizers.AdamOptimizer'])

        print(f"accuracy={acc:.4f}  f1_macro={f1:.4f}  run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()

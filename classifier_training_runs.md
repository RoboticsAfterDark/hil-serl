TODO(abhismirai42): Update the structure to be more intuitive

1) run for epochs = 200, training:test = 8:2
    ```
    Test Error: 
    Accuracy: 86.4%, Avg loss: 0.496386
    ```

2) run for epochs = 200, training:test = 7:3
    ````
    Test Error: 
    Accuracy: 90.9%, Avg loss: 0.279105
    ````

3) run for epochs = 500, training:test = 5:5
    ```
    Test Error: 
        Accuracy: 96.3%, Avg loss: 0.142616 
    ```

4) run for epochs = 100, training:test = 5:5
    ```
    [Epoch 100/100] Train loss: 0.005320, Test Accuracy: 94.4%, Test Avg loss: 0.172887

    Evaluation Accuracy: 97.2%, Evaluation loss: 0.088480
    Confusion matrix (rows: actual, columns: predicted):
                    failure     success
        failure          89           0
        success           3          16
    ```
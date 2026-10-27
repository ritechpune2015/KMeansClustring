using KMeansClustring.Models;
using Microsoft.ML;
int numberOfClusters = 3;

var mlContext = new MLContext(seed: 0);

Console.WriteLine("ML.NET K-Means Customer Segmentation");
Console.WriteLine("------------------------------------");

string dataPath = Path.Combine(AppContext.BaseDirectory, "customer.csv");

if (!File.Exists(dataPath))
{
    Console.WriteLine($"Dataset not found: {dataPath}");
    return;
}

IDataView data = mlContext.Data.LoadFromTextFile<CustomerData>(path: dataPath,
    hasHeader: true, separatorChar: ','
);
Console.WriteLine("Customer data loaded successfully.");


var pipeline =
    mlContext.Transforms.NormalizeMeanVariance(
        nameof(CustomerData.Income))
    .Append(
        mlContext.Transforms.NormalizeMeanVariance(
            nameof(CustomerData.Spending)))
    .Append(
        mlContext.Transforms.Concatenate(
            "Features",
            nameof(CustomerData.Income),
            nameof(CustomerData.Spending))
        .Append(
    mlContext.Clustering.Trainers.KMeans(
        featureColumnName: "Features",
        numberOfClusters: numberOfClusters)));

// Step 7: Train the model
Console.WriteLine("Training K-Means model...");


var model = pipeline.Fit(data);
Console.WriteLine("K-Means model trained successfully.");

var predictionEngine =
    mlContext.Model.CreatePredictionEngine<
        CustomerData,
        CustomerPrediction>(model);

// Step 9: Read customer records
var customers = mlContext.Data
    .CreateEnumerable<CustomerData>(
        data,
        reuseRowObject: false)
    .ToList();

// Step 10: Predict clusters for existing customers
Console.WriteLine("\nCustomer Cluster Assignments");
Console.WriteLine(
    "------------------------------------------------");

foreach (var customer in customers)
{
    var prediction = predictionEngine.Predict(customer);

    Console.WriteLine(
        $"{customer.CustomerId,-8} | " +
        $"Income: {customer.Income,5:F1} | " +
        $"Spending: {customer.Spending,5:F1} | " +
        $"Cluster: {prediction.PredictedCluster}");
}

var newCustomer = new CustomerData
{
    CustomerId = "NEW001",
    Income = 4.2f,
    Spending = 25f
};

var newPrediction = predictionEngine.Predict(newCustomer);

Console.WriteLine("\nNew Customer Prediction");
Console.WriteLine("-----------------------");
Console.WriteLine($"Customer ID: {newCustomer.CustomerId}");
Console.WriteLine($"Income: {newCustomer.Income:F1}");
Console.WriteLine($"Spending: {newCustomer.Spending:F1}");
Console.WriteLine(
    $"Predicted Cluster: {newPrediction.PredictedCluster}");

// Step 12: Display distances to centroids
Console.WriteLine("\nDistances to Cluster Centroids");

if (newPrediction.Distances is not null)
{
    for (int i = 0; i < newPrediction.Distances.Length; i++)
    {
        Console.WriteLine(
            $"Cluster {i + 1}: {newPrediction.Distances[i]:F4}");
    }
}

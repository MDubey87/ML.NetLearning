using Microsoft.ML;
using SentimentAnalysis;
using static Microsoft.ML.DataOperationsCatalog;

// <SnippetCreateMLContext>
MLContext mlContext = new MLContext();
// </SnippetCreateMLContext>

// <SnippetCallLoadData>
TrainTestData splitDataView = MLHelper.LoadData(mlContext);
// </SnippetCallLoadData>

// <SnippetCallBuildAndTrainModel>
ITransformer model = MLHelper.BuildAndTrainModel(mlContext, splitDataView.TrainSet);
// </SnippetCallBuildAndTrainModel>

// <SnippetCallEvaluate>
MLHelper.Evaluate(mlContext, model, splitDataView.TrainSet);
// </SnippetCallEvaluate>

// <SnippetCallUseModelWithSingleItem>
MLHelper.UseModelWithSingleItem(mlContext, model);
// </SnippetCallUseModelWithSingleItem>

// <SnippetCallUseModelWithBatchItems>
MLHelper.UseModelWithBatchItems(mlContext, model);
// </SnippetCallUseModelWithBatchItems>
Console.WriteLine();
Console.WriteLine("=============== End of process ===============");









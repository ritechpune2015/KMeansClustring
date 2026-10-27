using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace KMeansClustring.Models
{
    public class CustomerPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedCluster { get; set; }

        [ColumnName("Score")]
        public float[]? Distances { get; set; }

    }
}

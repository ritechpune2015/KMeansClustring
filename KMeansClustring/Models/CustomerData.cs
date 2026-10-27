using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace KMeansClustring.Models
{
    public class CustomerData
    {
        [LoadColumn(0)]
        public string CustomerId { get; set; } = string.Empty;
        [LoadColumn(1)]
        public float Income { get; set; }
        [LoadColumn(2)]
        public float Spending { get; set; }

    }
}

using System;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace EdgeMatchingSample
{
    public static class Program
    {
        private static readonly string ImagePath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory, "../../TestData/Test1");

        public static void Main()
        {
            using (var naturalOriginal = new Mat(Path.Combine(ImagePath, "natural.png"), ImreadModes.AnyColor))
            using (var natural = new Mat(naturalOriginal.Size(), MatType.CV_8UC3))
            using (var rendered = new Mat(Path.Combine(ImagePath, "rendered.png"), ImreadModes.AnyColor))
            using (var naturalEdge = new Mat(natural.Size(), MatType.CV_8UC1))
            using (var renderedEdge = new Mat(rendered.Size(), MatType.CV_8UC1))
            using (var naturalDescriptor = new Mat())
            using (var renderedDescriptor = new Mat())
            using (var akaze = AKAZE.Create())
            using (var resultMat = new Mat())
            using (var noiseRemovedContour = new Mat(natural.Size(), MatType.CV_8UC1))
            using (var matchingDebugMat = new Mat())
            using (var overlayBefore = new Mat())
            using (var overlayAfter = new Mat())
            {
                Cv2.BilateralFilter(naturalOriginal, natural, 9, 150, 150);

                //find edge
                using (var renderedMask = new Mat(rendered.Size(), MatType.CV_8UC1))
                using (var kernel = new Mat(new Size(21, 21), MatType.CV_8UC1, Scalar.White))
                using (var dilateMask = new Mat())
                using (var erosionMask = new Mat())
                {
                    Cv2.Threshold(rendered, renderedMask, 1, 255, ThresholdTypes.Binary);
                    Cv2.Canny(renderedMask, renderedEdge, 50, 50);

                    Cv2.FindContours(renderedEdge, out var renderedContour, out _, RetrievalModes.External,
                        ContourApproximationModes.ApproxSimple);
                    Cv2.DrawContours(renderedEdge, renderedContour, -1, Scalar.White, 3);

                    var channels = Cv2.Split(natural);

                    using (var buffer = new Mat())
                    {
                        foreach (var channel in channels)
                        {
                            Cv2.Canny(channel, buffer, 50, 50);
                            Cv2.BitwiseOr(buffer, naturalEdge, naturalEdge);
                            channel?.Dispose();
                        }
                    }

                    Cv2.Dilate(renderedMask, dilateMask, kernel);
                    Cv2.CvtColor(dilateMask, dilateMask, ColorConversionCodes.RGB2GRAY);
                    Cv2.BitwiseAnd(naturalEdge, dilateMask, naturalEdge);

                    Cv2.Erode(renderedMask, erosionMask, kernel);
                    Cv2.CvtColor(erosionMask, erosionMask, ColorConversionCodes.RGB2GRAY);
                    Cv2.Threshold(erosionMask, erosionMask, 128, 255, ThresholdTypes.BinaryInv);
                    Cv2.BitwiseAnd(naturalEdge, erosionMask, naturalEdge);

                    //remove noise from natural edge
                    Cv2.FindContours(naturalEdge, out var contours, out _, RetrievalModes.External,
                        ContourApproximationModes.ApproxSimple);
                    var continuousContour = contours.Where(item => Cv2.ContourArea(item) > 30);
                    Cv2.DrawContours(noiseRemovedContour, continuousContour, -1, Scalar.White, 3);
                }

                //akaze
                akaze.DetectAndCompute(noiseRemovedContour, null, out var keyPointsNatural, naturalDescriptor);
                akaze.DetectAndCompute(renderedEdge, null, out var keyPointsRendered, renderedDescriptor);

                //matching
                var matcher = new BFMatcher(NormTypes.Hamming, true);
                var matches = matcher.Match(naturalDescriptor, renderedDescriptor);

                var matchNumber = 100;
                var bestMatches = matches.Where(match => match.Distance < 100.0).Take(matchNumber).ToArray();

                var naturalPoints = bestMatches
                    .Select(match => keyPointsNatural[match.QueryIdx].Pt)
                    .Select(point => new Point2d(point.X, point.Y));
                var renderedPoints = bestMatches
                    .Select(match => keyPointsRendered[match.TrainIdx].Pt)
                    .Select(point => new Point2d(point.X, point.Y));

                Cv2.DrawMatches(noiseRemovedContour, keyPointsNatural, renderedEdge, keyPointsRendered, bestMatches, matchingDebugMat);

                //find homography
                var estimatedTransform = Cv2.FindHomography(naturalPoints, renderedPoints, HomographyMethods.Ransac);

                for (int row = 0; row < estimatedTransform.Rows; row++)
                {
                    for (int col = 0; col < estimatedTransform.Cols; col++)
                    {
                        Console.Write(estimatedTransform.Get<double>(row, col) + " ");
                    }
                    Console.WriteLine();
                }

                Cv2.WarpPerspective(natural, resultMat, estimatedTransform, natural.Size());

                //overlay debug
                Cv2.AddWeighted(rendered, 0.5, natural, 0.5, 0, overlayBefore);
                Cv2.AddWeighted(rendered, 0.5, resultMat, 0.5, 0, overlayAfter);

                //show
                Cv2.ImShow("natural", natural);
                Cv2.ImShow("rendered", rendered);
                Cv2.ImShow("natural edge", naturalEdge);
                Cv2.ImShow("rendered edge", renderedEdge);
                Cv2.ImShow("matching debug", matchingDebugMat);
                Cv2.ImShow("noise removed contour", noiseRemovedContour);
                Cv2.ImShow("result", resultMat);
                Cv2.ImShow("overlay before", overlayBefore);
                Cv2.ImShow("overlay after", overlayAfter);
                Cv2.WaitKey();
            }
        }
    }
}

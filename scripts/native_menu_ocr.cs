// Read-only image OCR worker. Pointer authorization stays in the Python native menu adapter.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using Windows.Foundation;
using Windows.Graphics.Imaging;
using Windows.Media.Ocr;
using Windows.Storage;

internal static class WindowsSavedImageOcr
{
    private static T Await<T>(IAsyncOperation<T> operation)
    {
        try
        {
            var timer = Stopwatch.StartNew();
            while (operation.Status == AsyncStatus.Started)
            {
                if (timer.ElapsedMilliseconds > 20000)
                {
                    operation.Cancel();
                    throw new TimeoutException("native saved-image operation exceeded its budget");
                }
                Thread.Sleep(10);
            }
            return operation.GetResults();
        }
        finally { operation.Close(); }
    }

    private static string Hash(string path)
    {
        using (var stream = File.OpenRead(path))
        using (var algorithm = SHA256.Create())
            return BitConverter.ToString(algorithm.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
    }

    private static string Inside(string root, string path)
    {
        string full = Path.GetFullPath(path);
        string prefix = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        if (!full.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("saved input and new output must be inside the supplied workspace");
        return full;
    }

    private static Dictionary<string, object> Capabilities()
    {
        var languages = new List<string>();
        foreach (var language in OcrEngine.AvailableRecognizerLanguages)
            languages.Add(language.LanguageTag);
        return new Dictionary<string, object> {
            {"kind", "windows_native_ocr_saved_image"}, {"completed", false},
            {"available_languages", languages}, {"max_image_dimension", OcrEngine.MaxImageDimension},
            {"external_model_calls", 0}, {"environment_inputs", 0}, {"vm_started", false},
            {"evaluator_calls", 0}, {"ocr_recognition_calls", 0},
            {"api_reference", "https://learn.microsoft.com/en-us/uwp/api/windows.media.ocr.ocrengine"},
            {"limitations", new [] {
                "Read-only saved image; recognized words do not establish input capability or task effect.",
                "Installed native recognizer only; no language download, dependency install or host policy changes.",
                "Recognized text alone never authorizes UI input; the caller must rebind native menu evidence."
            }}
        };
    }

    private static string Json(object value)
    {
        return new JavaScriptSerializer().Serialize(value);
    }

    private static Dictionary<string, object> Recognize(string input, string output, OcrEngine engine)
    {
        var report = Capabilities();
        report["source_image_sha256"] = Hash(input);
        report["helper_binary_sha256"] = Hash(typeof(WindowsSavedImageOcr).Assembly.Location);
        try
        {
            StorageFile file = Await(StorageFile.GetFileFromPathAsync(input));
            using (var stream = Await(file.OpenAsync(FileAccessMode.Read)))
            {
                var decoder = Await(BitmapDecoder.CreateAsync(stream));
                if (decoder.PixelWidth == 0 || decoder.PixelHeight == 0 ||
                    decoder.PixelWidth > OcrEngine.MaxImageDimension ||
                    decoder.PixelHeight > OcrEngine.MaxImageDimension)
                    throw new ArgumentException("image dimensions exceed native recognizer budget");
                using (var bitmap = Await(decoder.GetSoftwareBitmapAsync()))
                using (var converted = SoftwareBitmap.Convert(bitmap, BitmapPixelFormat.Bgra8,
                                                               BitmapAlphaMode.Premultiplied))
                {
                    if (engine == null)
                        throw new InvalidOperationException("no installed recognizer is available");
                    report["recognizer_language"] = engine.RecognizerLanguage.LanguageTag;
                    report["width"] = converted.PixelWidth;
                    report["height"] = converted.PixelHeight;
                    var timer = Stopwatch.StartNew();
                    report["ocr_recognition_calls"] = 1;
                    var result = Await(engine.RecognizeAsync(converted));
                    timer.Stop();
                    report["recognize_ms"] = timer.Elapsed.TotalMilliseconds;
                    var lines = new List<object>();
                    int count = 0;
                    foreach (var line in result.Lines)
                    {
                        var words = new List<object>();
                        foreach (var word in line.Words)
                        {
                            if (++count > 2000 || word.Text.Length > 2000)
                                throw new InvalidOperationException("recognized text exceeds diagnostic budget");
                            var box = word.BoundingRect;
                            words.Add(new Dictionary<string, object> {
                                {"text", word.Text}, {"x", box.X}, {"y", box.Y},
                                {"width", box.Width}, {"height", box.Height}
                            });
                        }
                        lines.Add(new Dictionary<string, object> {{"text", line.Text}, {"words", words}});
                    }
                    report["lines"] = lines;
                    report["word_count"] = count;
                    report["completed"] = true;
                }
            }
        }
        catch (Exception error)
        {
            report["error_type"] = error.GetType().FullName;
        }
        Write(output, report);
        return report;
    }

    private static void Write(string output, object report)
    {
        using (var stream = new FileStream(output, FileMode.CreateNew, FileAccess.Write))
        using (var writer = new StreamWriter(stream, new UTF8Encoding(false)))
            writer.WriteLine(Json(report));
    }

    private static void CheckPaths(string input, string output)
    {
        if (!File.Exists(input) || new FileInfo(input).Length > 24000000 || File.Exists(output) ||
            Directory.Exists(output))
            throw new ArgumentException("bounded existing input and unused output required");
    }

    private static int Batch(string root, string input, string output)
    {
        if (new FileInfo(input).Length > 65536)
            throw new ArgumentException("batch manifest exceeds budget");
        var manifest = new JavaScriptSerializer().DeserializeObject(File.ReadAllText(input))
                       as Dictionary<string, object>;
        if (manifest == null || manifest.Count != 2 ||
            !manifest.ContainsKey("kind") || !manifest.ContainsKey("tiles") ||
            !Equals(manifest["kind"], "native_menu_ocr_batch_v1"))
            throw new ArgumentException("invalid batch manifest");
        var tiles = manifest["tiles"] as object[];
        if (tiles == null || tiles.Length < 1 || tiles.Length > 16)
            throw new ArgumentException("invalid batch tile count");
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {input, output};
        var inputs = new List<string>();
        var outputs = new List<string>();
        long bytes = 0;
        foreach (var entry in tiles)
        {
            var tile = entry as Dictionary<string, object>;
            if (tile == null || tile.Count != 3 || !tile.ContainsKey("input") ||
                !tile.ContainsKey("output") || !tile.ContainsKey("source_image_sha256") ||
                !(tile["input"] is string) || !(tile["output"] is string))
                throw new ArgumentException("invalid batch tile");
            string source = Inside(root, (string)tile["input"]);
            string receipt = Inside(root, (string)tile["output"]);
            if (!paths.Add(source) || !paths.Add(receipt))
                throw new ArgumentException("batch paths must be unique");
            CheckPaths(source, receipt);
            bytes += new FileInfo(source).Length;
            if (bytes > 128000000 || !Equals(tile["source_image_sha256"], Hash(source)))
                throw new ArgumentException("batch input hash or byte budget mismatch");
            inputs.Add(source);
            outputs.Add(receipt);
        }
        var report = new Dictionary<string, object> {
            {"kind", "windows_native_ocr_saved_batch_v1"}, {"completed", false},
            {"source_manifest_sha256", Hash(input)},
            {"helper_binary_sha256", Hash(typeof(WindowsSavedImageOcr).Assembly.Location)},
            {"environment_inputs", 0}, {"external_model_calls", 0}, {"ocr_recognition_calls", 0}
        };
        var proofs = new List<object>();
        report["tiles"] = proofs;
        var timer = Stopwatch.StartNew();
        try
        {
            var engine = OcrEngine.TryCreateFromUserProfileLanguages();
            if (engine == null)
                throw new InvalidOperationException("no installed recognizer is available");
            int calls = 0;
            for (int index = 0; index < inputs.Count; ++index)
            {
                var tileReport = Recognize(inputs[index], outputs[index], engine);
                calls += (int)tileReport["ocr_recognition_calls"];
                report["ocr_recognition_calls"] = calls;
                proofs.Add(new Dictionary<string, object> {
                    {"output", outputs[index]}, {"source_image_sha256", tileReport["source_image_sha256"]},
                    {"receipt_sha256", Hash(outputs[index])}
                });
                if (!(bool)tileReport["completed"])
                    throw new InvalidOperationException("batch tile recognition failed");
            }
            report["completed"] = true;
        }
        catch (Exception error) { report["error_type"] = error.GetType().FullName; }
        report["wall_ms"] = timer.Elapsed.TotalMilliseconds;
        Write(output, report);
        Console.WriteLine(Json(new Dictionary<string, object> {
            {"completed", report["completed"]}, {"ocr_recognition_calls", report["ocr_recognition_calls"]}
        }));
        return (bool)report["completed"] ? 0 : 2;
    }

    public static int Main(string[] args)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        if (args.Length == 1 && args[0] == "--capabilities")
        {
            var capabilities = Capabilities();
            capabilities["completed"] = true;
            capabilities["batch_layout"] = "frame_batch_v1";
            Console.WriteLine(Json(capabilities));
            return 0;
        }
        if (args.Length != 4 || (args[0] != "--recognize-saved" && args[0] != "--recognize-saved-batch"))
        {
            Console.WriteLine("Use --capabilities or --recognize-saved[-batch] WORKSPACE INPUT NEW_OUTPUT");
            return 2;
        }
        string input = Inside(args[1], args[2]);
        string output = Inside(args[1], args[3]);
        CheckPaths(input, output);
        if (args[0] == "--recognize-saved-batch")
            return Batch(args[1], input, output);
        var report = Recognize(input, output, OcrEngine.TryCreateFromUserProfileLanguages());
        Console.WriteLine(Json(new Dictionary<string, object> {
            {"completed", report["completed"]}, {"ocr_recognition_calls", report["ocr_recognition_calls"]}
        }));
        return (bool)report["completed"] ? 0 : 2;
    }
}

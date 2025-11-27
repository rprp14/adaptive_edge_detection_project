// Fetch latest uploaded image and processed outputs
/*window.onload = function () {
  fetch("http://127.0.0.1:5000/get_latest_image")
    .then((response) => response.json())
    .then((data) => {
      console.log("Fetched data:", data);

      document.getElementById("uploadedImage").src = data.file_path;
      document.getElementById("basicOutput").src = data.basic_output;
      document.getElementById("adaptiveOutput").src = data.adaptive_output;
      document.getElementById("sobelOutput").src = data.sobel_output;
      document.getElementById("psoOutput").src = data.pso_output;
      document.getElementById("mlOutput").src = data.ml_output;
      document.getElementById("vitOutput").src = data.vit_output;
      document.getElementById("vitPsoOutput").src = data.vit_pso_output;
    })
    .catch((error) => console.error("Error fetching image:", error));
};*/

//========================================================================================================================================================

// Handle image upload and display all outputs + accuracy
document.getElementById("uploadForm").addEventListener("submit", function (e) {
  e.preventDefault();

  let fileInput = document.getElementById("fileInput");
  if (fileInput.files.length === 0) {
    alert("Please select an image.");
    return;
  }

  let formData = new FormData();
  formData.append("file", fileInput.files[0]);

  fetch("http://127.0.0.1:5000/upload", {
    method: "POST",
    body: formData
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Server Response:", data);

      // Display all images
      document.getElementById("uploadedImage").src = data.original_url;

      document.getElementById("basicOutput").src = data.basic_output;
      document.getElementById("sobelOutput").src = data.sobel_output;
      document.getElementById("adaptiveOutput").src = data.adaptive_output;
      document.getElementById("psoOutput").src = data.pso_output;
      document.getElementById("mlOutput").src = data.dl_output;
      document.getElementById("vitOutput").src = data.vit_output;
      document.getElementById("vitPsoOutput").src = data.vit_pso_output;

      // ===============================
      // ADD ACCURACY BELOW EACH IMAGE
      // ===============================

      document.getElementById("basicAccuracy").innerHTML =
        "Accuracy: " + data.basic_accuracy + "%";

      document.getElementById("sobelAccuracy").innerHTML =
        "Accuracy: " + data.sobel_accuracy + "%";

      document.getElementById("adaptiveAccuracy").innerHTML =
        "Accuracy: " + data.adaptive_accuracy + "%";

      document.getElementById("psoAccuracy").innerHTML =
        "Accuracy: " + data.pso_accuracy + "%";

      document.getElementById("mlAccuracy").innerHTML =
        "Accuracy: " + data.dl_accuracy + "%";

      document.getElementById("vitAccuracy").innerHTML =
        "Accuracy: " + data.vit_accuracy + "%";

      document.getElementById("vitPsoAccuracy").innerHTML =
        "Accuracy: " + data.vit_pso_accuracy + "%";
    })
    .catch((error) => {
      console.error("Error uploading:", error);
      alert("Upload failed.");
    });
});


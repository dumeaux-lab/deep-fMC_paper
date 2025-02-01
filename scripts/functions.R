##### draw heatmap #####
# Load required library
library(pheatmap)
# Define the function to draw the heatmap and save as PDF
draw_heatmap <- function(data, metadata, disease_colors, study_colors, pdf_path, width = 10, height = 10) {
  
  # Columns to scale
  columns_to_scale <- c("type1", "type2", "type3")
  
  # Function to scale to [0,1]
  scale_to_01 <- function(x) {
    (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  }
  
  # Create a scaled version of the metadata for color mapping only
  metadata_scaled_for_colors <- metadata
  #metadata_scaled_for_colors[columns_to_scale] <- lapply(metadata_scaled_for_colors[columns_to_scale], scale_to_01)
  
  # Define fixed gradient colors for scaled annotations
  type_colors_fixed <- list(
    type1 = colorRampPalette(c("blue", "white", "red"))(100),
    type2 = colorRampPalette(c("blue", "white", "red"))(100),
    type3 = colorRampPalette(c("blue", "white", "red"))(100)
  )
  
  # Combine annotation colors
  annotation_colors_combined <- c(
    list(disease = disease_colors, study = study_colors),
    type_colors_fixed
  )
  annotation_col_scale <- list(
    type1 = seq(0, 1, length.out = 101),
    type2 = seq(0, 1, length.out = 101),
    type3 = seq(0, 1, length.out = 101)
  )
  
  # Open PDF device
  pdf(pdf_path, width = width, height = height)
  heatmap_colors <- colorRampPalette(c("blue", "white", "orange"))(100)
  max_val <- max(data, na.rm = TRUE)
  min_val <- min(data, na.rm = TRUE)
  #breaks <- c(seq(min_val, 0, length.out = 51), seq(0, max_val, length.out = 50)[-1])
  breaks = seq(-1, 1, length.out = 101)  
  # Draw the heatmap with scaled metadata
  pheatmap(
    data, 
    cluster_rows = FALSE, 
    cluster_cols = FALSE, 
    show_rownames = TRUE, 
    show_colnames = TRUE,
    fontsize_row = 5,
    fontsize = 5.5,
    annotation_col = metadata_scaled_for_colors,
    annotation_colors = annotation_colors_combined,
    annotation_breaks = annotation_col_scale,  
    legend = TRUE,
    color = heatmap_colors,# Set color scale for heatmap values
    breaks = breaks  # Ensure gradient is centered at 0
  )
  
  # Close the PDF device
  dev.off()
  
  cat("Heatmap saved to", pdf_path, "\n")
}


##### get dominant type #####
get_dominant_type <- function(row) {
  # Replace this with the logic you want to use to determine the dominant type
  # For example, assuming type1, type2, type3 columns exist in metadata_row:
  max_type <- names(row)[which.max(row[c("type1", "type2", "type3")])]
  return(max_type)
}

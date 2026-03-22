from datetime import datetime
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import torch


def save_matrix_image(
	N: torch.Tensor,
	itos: Mapping[int, str] | None = None,
	output_dir: str = "graphs",
) -> Path:
	N_plot = N.detach().cpu()
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	image_path = output_path / f"N_{timestamp}.png"

	plt.figure(figsize=(16, 16))
	plt.imshow(N_plot.numpy(), cmap="Blues")

	rows, cols = N_plot.shape
	for i in range(rows):
		for j in range(cols):
			if itos is not None:
				chstr = f"{itos.get(i, '?')}{itos.get(j, '?')}"
				cell_text = f"{chstr}\n{N_plot[i, j].item():.2f}"
			else:
				cell_text = str(N_plot[i, j].item())
			plt.text(j, i, cell_text, ha="center", va="center", color="gray")

	plt.axis("off")
	plt.tight_layout()
	plt.savefig(image_path, dpi=150)
	plt.close()

	return image_path


def save_matrix_image_default(
	N: torch.Tensor,
	output_dir: str = "graphs",
) -> Path:
	N_plot = N.detach().cpu()
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	image_path = output_path / f"N_default_{timestamp}.png"

	plt.figure()
	plt.imshow(N_plot.numpy(), cmap="Blues")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(image_path, dpi=150)
	plt.close()

	return image_path

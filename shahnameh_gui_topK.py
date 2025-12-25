import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys

# Optional bidi/reshaper
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_BIDI = True
except Exception:
    _HAS_BIDI = False

# Paths (adjust if needed)
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "shahnameh_model_weights.pt")
TOKEN_PATH = os.path.join(BASE_DIR, "shahnameh_tokenizer.pkl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Build model and load weights
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

def load_model_and_tokenizer(weights_path=WEIGHTS_PATH, token_path=TOKEN_PATH):
    # Load PyTorch weights
    checkpoint = torch.load(weights_path, map_location=device)
    
    with open(token_path, 'rb') as f:
        tokens = pickle.load(f)

    vocab_size = len(tokens)
    # Extract dimensions from checkpoint
    embedding_dim = checkpoint['embedding.weight'].shape[1]
    hidden_dim = checkpoint['gru.weight_hh_l0'].shape[0] // 3  # GRU has 3 gates

    model = MyModel(vocab_size, embedding_dim, hidden_dim).to(device)
    model.eval()

    # Load state_dict directly
    model.load_state_dict(checkpoint)

    char_to_id = {c: i for i, c in enumerate(tokens)}
    id_to_char = {i: c for i, c in enumerate(tokens)}

    return model, char_to_id, id_to_char, hidden_dim

def sample_logits(logits, top_k=12, temperature=1.0):
    top_k = min(top_k, len(logits))
    topk_idx = np.argpartition(-logits, top_k-1)[:top_k]
    topk_vals = logits[topk_idx] / temperature
    exp = np.exp(topk_vals - np.max(topk_vals))
    p = exp / exp.sum()
    return int(np.random.choice(topk_idx, p=p))

def generate_text(model, char_to_id, id_to_char, hidden_dim, prompt, max_length=250, temperature=0.8, top_k=12):
    model.eval()
    start_ids = [char_to_id.get(c, 1) for c in prompt]
    state = torch.zeros(1,1,hidden_dim).to(device)
    with torch.no_grad():
        for token in start_ids:
            x = torch.tensor([[token]]).to(device)
            _, state = model(x, state)
        generated = []
        cur = start_ids[-1]
        for _ in range(max_length):
            inp = torch.tensor([[cur]]).to(device)
            out, state = model(inp, state)
            logits = out[0,0].detach().cpu().numpy()
            nid = sample_logits(logits, top_k=top_k, temperature=temperature)
            generated.append(nid)
            cur = nid
    return ''.join(id_to_char.get(i, '') for i in generated)

class ShahnamehGUI:
    def __init__(self, root):
        self.root = root
        root.title('Shahnameh Generator')

        frame = ttk.Frame(root, padding=8)
        frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frame, text='Prompt:').grid(row=0, column=0, sticky='w')
        self.prompt_entry = tk.Text(frame, height=4, width=60)
        self.prompt_entry.grid(row=1, column=0, columnspan=3, sticky='we')

        ttk.Label(frame, text='Max length:').grid(row=2, column=0, sticky='w')
        self.max_entry = ttk.Entry(frame, width=8)
        self.max_entry.insert(0, '200')
        self.max_entry.grid(row=2, column=1, sticky='w')

        ttk.Label(frame, text='Temperature:').grid(row=2, column=2, sticky='w')
        self.temp_entry = ttk.Entry(frame, width=8)
        self.temp_entry.insert(0, '0.8')
        self.temp_entry.grid(row=2, column=3, sticky='w')

        ttk.Label(frame, text='Top-k:').grid(row=2, column=4, sticky='w')
        self.topk_entry = ttk.Entry(frame, width=8)
        self.topk_entry.insert(0, '12')
        self.topk_entry.grid(row=2, column=5, sticky='w')

        self.gen_button = ttk.Button(frame, text='Generate', command=self.on_generate)
        self.gen_button.grid(row=3, column=0, pady=8)

        self.save_button = ttk.Button(frame, text='Save HTML', command=self.on_save)
        self.save_button.grid(row=3, column=1, pady=8)

        self.output = scrolledtext.ScrolledText(frame, height=20, width=80, wrap='word')
        self.output.grid(row=4, column=0, columnspan=6)

        self.status = ttk.Label(frame, text='Loading model...')
        self.status.grid(row=5, column=0, columnspan=6, sticky='w')

        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.model, self.char_to_id, self.id_to_char, self.hidden_dim = load_model_and_tokenizer()
            self.status.config(text='Model loaded.')
        except Exception as e:
            self.status.config(text=f'Failed to load model: {e}')
            messagebox.showerror('Error', f'Failed to load model:\n{e}')

    def on_generate(self):
        prompt = self.prompt_entry.get('1.0', 'end').strip()
        if not prompt:
            messagebox.showinfo('Info', 'Please enter a prompt')
            return
        try:
            max_len = int(self.max_entry.get())
        except Exception:
            max_len = 200
        try:
            temp = float(self.temp_entry.get())
        except Exception:
            temp = 0.8
        try:
            topk = int(self.topk_entry.get())
        except Exception:
            topk = 12

        self.gen_button.config(state='disabled')
        self.status.config(text='Generating...')
        threading.Thread(target=self._generate_thread, args=(prompt, max_len, temp, topk), daemon=True).start()

    def _generate_thread(self, prompt, max_len, temp, topk):
        try:
            out = generate_text(self.model, self.char_to_id, self.id_to_char, self.hidden_dim, prompt, max_length=max_len, temperature=temp, top_k=topk)
            full = prompt + out
            
            # Apply RTL processing while preserving newlines
            if _HAS_BIDI:
                try:
                    # Process each line separately to preserve newlines
                    lines = full.split('\n')
                    processed_lines = []
                    for line in lines:
                        reshaped = arabic_reshaper.reshape(line)
                        display_line = get_display(reshaped)
                        processed_lines.append(display_line)
                    display = '\n'.join(processed_lines)
                except Exception:
                    display = full
            else:
                # Process each line separately for RTL with newlines preserved
                lines = full.split('\n')
                rtl_lines = [line[::-1] for line in lines]
                display = '\n'.join(rtl_lines)
            
            self.output.delete('1.0', 'end')
            self.output.insert('end', display)
            self.status.config(text='Done')
        except Exception as e:
            self.status.config(text=f'Error: {e}')
            messagebox.showerror('Generation error', str(e))
        finally:
            self.gen_button.config(state='normal')

    def on_save(self):
        txt = self.output.get('1.0', 'end').strip()
        if not txt:
            messagebox.showinfo('Info', 'Nothing to save')
            return
        fname = filedialog.asksaveasfilename(defaultextension='.html', filetypes=[('HTML files','*.html')])
        if not fname:
            return
        try:
            html = f"""
            <!doctype html>
            <html><head><meta charset='utf-8'><title>Shahnameh Output</title></head>
            <body style='direction: rtl; font-family: "Segoe UI", "Tahoma", sans-serif; white-space: pre-wrap;'>
            {txt}
            </body></html>
            """
            with open(fname, 'w', encoding='utf-8') as fh:
                fh.write(html)
            messagebox.showinfo('Saved', f'Saved to {fname}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = ShahnamehGUI(root)
    root.mainloop()
import tkinter as tk
from tkinter import ttk, messagebox
import re
import unicodedata

# Helper functions for LaTeX math processing and normalization
def clean_latex_formatting(text):
    # Fix backslash spacing: "\ alpha" -> "\alpha"
    text = re.sub(r'\\\s+([a-zA-Z]+)', r'\\\1', text)
    
    # Fix spacing around subscripts and superscripts: "x _ i" -> "x_i", "N ^ 2" -> "N^2"
    text = re.sub(r'(\w+|\\?\w+)\s*_\s*(\w+|\{[^{}]+\})', r'\1_\2', text)
    text = re.sub(r'(\w+|\\?\w+)\s*\^\s*(\w+|\{[^{}]+\})', r'\1^\2', text)
    
    # Fix spaces inside braces of subscripts and superscripts: "_{ t - 1 }" -> "_{t-1}"
    text = re.sub(r'_(?:\{([^}]+)\})', lambda m: '_{' + m.group(1).replace(' ', '') + '}', text)
    text = re.sub(r'\^(?:\{([^}]+)\})', lambda m: '^{' + m.group(1).replace(' ', '') + '}', text)
    
    # Standardize complexities: "O ( N 2 )" or "O ( N )" or "O ( 1 )"
    text = re.sub(r'[oO]\s*\(\s*N\s*(?:\^?\s*2|²)?\s*\)', 'O(N^2)', text)
    text = re.sub(r'[oO]\s*\(\s*N\s*\)', 'O(N)', text)
    text = re.sub(r'[oO]\s*\(\s*1\s*\)', 'O(1)', text)
    return text

def wrap_math_in_delimiters(text):
    placeholders = []
    def store_placeholder(m):
        placeholders.append(m.group(0))
        return f"__MATH_PH_{len(placeholders)-1}__"
        
    # Store existing $...$ and $$...$$
    text = re.sub(r'\$\$.*?\$\$', store_placeholder, text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', store_placeholder, text)
    
    # Define nested braces pattern (up to 2 levels)
    nested_braces = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}'
    
    # 1. Wrap inline equations: e.g., X_{seasonal} = X - X_{trend} or X_{\text{trend}} = \text{AvgPool1d}(X, \text{kernel\_size})
    lhs_part = r'\b[a-zA-Z0-9_+\-*\/()\[\]{}^\\,α-ωΑ-Ω]+(?:\s*_\s*(?:[a-zA-Z0-9]+|' + nested_braces + r'))?'
    relation_part = r'\s*(?:=|\\approx|\\le|\\ge|<|>)\s*'
    rhs_part = r'(?:[a-zA-Z0-9_+\-*\/()\[\]{}^\\,α-ωΑ-Ω]+|\s+(?=[-+*\/()\[\]{}^\\,=<>])|(?<=[-+$*\/()\[\]{}^\\,=<>])\s+)+'
    
    inline_eq_regex = lhs_part + relation_part + rhs_part
    text = re.sub(inline_eq_regex, lambda m: f"${m.group(0)}$", text)
    
    # Store newly wrapped inline equations as placeholders
    text = re.sub(r'\$.*?\$', store_placeholder, text)
    
    # 2. Wrap standalone O(N^2), O(N), O(1)
    text = re.sub(r'\b(O\(N\^2\)|O\(N\)|O\(1\))\b', r'$\1$', text)
    
    # 3. Wrap subscripted variables (supporting nested braces): e.g., h_t, h_{t-1}, w_c, \alpha_c, \Delta_t, d_k
    math_var_pattern = r'\b([a-zA-Z]|\\(?:alpha|beta|gamma|delta|Delta|epsilon|theta|lambda|mu|nu|xi|pi|rho|sigma|Sigma|tau|phi|omega|eta|chi|psi))\s*_\s*([a-zA-Z0-9]+|' + nested_braces + r')'
    text = re.sub(math_var_pattern, r'$\1_\2$', text)
    
    # 4. Wrap standalone greek letters that are not already wrapped:
    greek_pattern = r'\b\\(alpha|beta|gamma|delta|Delta|epsilon|theta|lambda|mu|nu|xi|pi|rho|sigma|Sigma|tau|phi|omega|eta|chi|psi)\b'
    text = re.sub(greek_pattern, r'$\1$', text)
    
    # Restore placeholders
    for i, ph in enumerate(placeholders):
        text = text.replace(f"__MATH_PH_{i}__", ph)
        
    return text

def wrap_block_equations_logic(text):
    placeholders = []
    def store_placeholder(m):
        placeholders.append(m.group(0))
        return f"__MATH_PH_{len(placeholders)-1}__"
        
    text = re.sub(r'\$\$.*?\$\$', store_placeholder, text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', store_placeholder, text)
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if not line_strip:
            continue
        has_eq = '=' in line_strip
        has_latex = '\\' in line_strip or '_' in line_strip or '^' in line_strip
        has_math_op = any(op in line_strip for op in ['+', '-', '*', '/', '<', '>', '\\approx', '\\le', '\\ge', '\\in'])
        vietnamese_chars = re.findall(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', line_strip, re.IGNORECASE)
        
        if has_eq and (has_latex or has_math_op) and len(vietnamese_chars) == 0:
            lines[i] = f"$${line_strip}$$"
            
    text = '\n'.join(lines)
    for i, ph in enumerate(placeholders):
        text = text.replace(f"__MATH_PH_{i}__", ph)
    return text

def apply_bold_italic_formatting(text):
    # Accents
    text = re.sub(r'\\hat\{([^{}]+)\}', lambda m: m.group(1) + '\u0302', text)
    text = re.sub(r'\\bar\{([^{}]+)\}', lambda m: m.group(1) + '\u0304', text)
    text = re.sub(r'\\tilde\{([^{}]+)\}', lambda m: m.group(1) + '\u0303', text)
    text = re.sub(r'\\vec\{([^{}]+)\}', lambda m: m.group(1) + '\u20D7', text)
    
    # LaTeX bold/italic nested -> Markdown bold-italic
    text = re.sub(r'\\(?:textit|mathit)\{\s*\\(?:textbf|mathbf|bm)\{([^{}]+)\}\s*\}', r'***\1***', text)
    text = re.sub(r'\\(?:textbf|mathbf|bm)\{\s*\\(?:textit|mathit)\{([^{}]+)\}\s*\}', r'***\1***', text)
    
    # LaTeX bold -> Markdown bold
    text = re.sub(r'\\(?:textbf|mathbf|bm)\{([^{}]+)\}', r'**\1**', text)
    
    # LaTeX italic -> Markdown italic
    text = re.sub(r'\\(?:textit|mathit)\{([^{}]+)\}', r'*\1*', text)
    
    # LaTeX Roman
    text = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', text)
    
    return text

def convert_markdown_bold_italic(text):
    # We keep markdown styling in plain text so it can be copied cleanly as rich HTML format.
    return text

def markdown_to_html(text):
    import html as html_module
    
    # 1. Extract math blocks and convert them to Cambria Math spans, storing them as placeholders
    placeholders = []
    def store_math_placeholder(m):
        formula = m.group(1)
        # Escape HTML special chars inside the formula (e.g. <, >, &)
        escaped_formula = html_module.escape(formula)
        span = f'<span style="font-family: \'Cambria Math\', serif; font-size: 13pt; background-color: transparent;">{escaped_formula}</span>'
        placeholders.append(span)
        # Use an alphanumeric placeholder format with no underscores or asterisks to avoid Markdown match
        return f"PHMATHSPAN{len(placeholders)-1}X"
        
    # We must match $$ first, then $
    text = re.sub(r'\$\$(.*?)\$\$', store_math_placeholder, text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', store_math_placeholder, text)
    
    # 2. Escape HTML special characters for the non-math text
    html = html_module.escape(text)
    
    # 3. Convert Markdown formatting to HTML tags
    html = re.sub(r'\*\*\*([^*]+)\*\*\*', r'<b><i>\1</i></b>', html)
    html = re.sub(r'___([^_]+)___', r'<b><i>\1</i></b>', html)
    
    html = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', html)
    html = re.sub(r'__([^_]+)__', r'<b>\1</b>', html)
    
    html = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', html)
    html = re.sub(r'_([^_]+)_', r'<i>\1</i>', html)
    
    # Convert newlines to HTML line breaks
    html = html.replace('\n', '<br>\n')
    
    # 4. Restore the math placeholders
    for i, ph in enumerate(placeholders):
        html = html.replace(f"PHMATHSPAN{i}X", ph)
        
    return html

def set_clipboard_html_and_text(html_content, plain_text):
    try:
        import ctypes
        from ctypes import wintypes
        
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        
        # 1. Prepare HTML Clipboard format payload
        header_template = (
            "Version:0.9\r\n"
            "StartHTML:{:010d}\r\n"
            "EndHTML:{:010d}\r\n"
            "StartFragment:{:010d}\r\n"
            "EndFragment:{:010d}\r\n"
        )
        
        dummy_header = header_template.format(0, 0, 0, 0)
        dummy_header_bytes_len = len(dummy_header.encode('utf-8'))
        
        html_prefix = '<html>\r\n<body style="font-family: \'Times New Roman\', serif; font-size: 13pt; background-color: transparent;">\r\n<!--StartFragment-->'
        html_suffix = "<!--EndFragment-->\r\n</body>\r\n</html>"
        
        prefix_bytes_len = len(html_prefix.encode('utf-8'))
        content_bytes_len = len(html_content.encode('utf-8'))
        suffix_bytes_len = len(html_suffix.encode('utf-8'))
        
        start_html = dummy_header_bytes_len
        start_fragment = start_html + prefix_bytes_len
        end_fragment = start_fragment + content_bytes_len
        end_html = end_fragment + suffix_bytes_len
        
        final_header = header_template.format(start_html, end_html, start_fragment, end_fragment)
        final_html_payload = final_header + html_prefix + html_content + html_suffix
        final_html_bytes = final_html_payload.encode('utf-8')
        
        # 2. Prepare Plain Text payload (UTF-16 LE null-terminated)
        plain_text_bytes = (plain_text + '\x00').encode('utf-16le')
        
        # 3. Open Clipboard
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        if not user32.OpenClipboard(None):
            return False
            
        user32.EmptyClipboard.argtypes = []
        user32.EmptyClipboard.restype = wintypes.BOOL
        user32.EmptyClipboard()
        
        # Set plain text
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        h_text = kernel32.GlobalAlloc(0x0002, len(plain_text_bytes)) # GMEM_MOVEABLE = 0x0002
        if h_text:
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            p_text = kernel32.GlobalLock(h_text)
            if p_text:
                ctypes.memmove(p_text, plain_text_bytes, len(plain_text_bytes))
                kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
                kernel32.GlobalUnlock.restype = wintypes.BOOL
                kernel32.GlobalUnlock(h_text)
                user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
                user32.SetClipboardData.restype = ctypes.c_void_p
                user32.SetClipboardData(13, h_text) # 13 = CF_UNICODETEXT
                
        # Set HTML
        user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterClipboardFormatW.restype = wintypes.UINT
        cf_html = user32.RegisterClipboardFormatW("HTML Format")
        if cf_html:
            h_html = kernel32.GlobalAlloc(0x0002, len(final_html_bytes) + 1)
            if h_html:
                p_html = kernel32.GlobalLock(h_html)
                if p_html:
                    ctypes.memmove(p_html, final_html_bytes, len(final_html_bytes))
                    ctypes.memset(p_html + len(final_html_bytes), 0, 1) # Null-terminator
                    kernel32.GlobalUnlock(h_html)
                    user32.SetClipboardData(cf_html, h_html)
                    
        user32.CloseClipboard()
        return True
    except Exception as e:
        print(f"Error writing HTML to clipboard: {e}")
    return False

def clean_math_syntax(math_text):
    # Strip \text{...} and \mathrm{...}
    for _ in range(3):
        math_text = re.sub(r'\\(?:text|mathrm)\{([^{}]+)\}', r'\1', math_text)
    # Replace escaped underscores
    math_text = math_text.replace(r'\_', '_')
    # Clean leftover LaTeX commands (like \left, \right)
    math_text = re.sub(r'\\(?:left|right)', '', math_text)
    return math_text

def process_math_content(math_text):
    # First clean basic math syntax
    math_text = clean_math_syntax(math_text)
    
    # Greek letter map
    greek_map = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', r'\Delta': 'Δ',
        r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ', r'\iota': 'ι',
        r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
        r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ', r'\Sigma': 'Σ', r'\tau': 'τ',
        r'\upsilon': 'υ', r'\phi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
        r'\Omega': 'Ω', r'\theta': 'θ', r'\Theta': 'Θ'
    }
    # Math operators (removed \text)
    math_map = {
        r'\times': '×', r'\cdot': '·', r'\approx': '≈', r'\le': '≤', r'\ge': '≥',
        r'\in': '∈', r'\notin': '∉', r'\neq': '≠', r'\infty': '∞', r'\propto': '∝',
        r'\partial': '∂', r'\nabla': '∇', r'\forall': '∀', r'\exists': '∃',
        r'\emptyset': 'Ø', r'\sum': '∑', r'\int': '∫', r'\sqrt': '√',
        r'\mathbb{R}': 'ℝ', r'\mathbb{C}': 'ℂ', r'\mathbb{Z}': 'ℤ', r'\mathbb{N}': 'ℕ',
        r'\lfloor': '⌊', r'\rfloor': '⌋'
    }
    
    for lat, uni in greek_map.items():
        math_text = math_text.replace(lat, uni)
    for lat, uni in math_map.items():
        math_text = math_text.replace(lat, uni)
        
    super_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', 
                 '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾', 'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'T': 'ᵀ'}
    sub_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉', 
               '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎', 'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ', 
               'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 
               'v': 'ᵥ', 'x': 'ₓ'}
    
    def rep_super(m):
        content = m.group(1) or m.group(2)
        content = content.strip()
        return "".join(super_map.get(c, '^' + c) for c in content)
        
    math_text = re.sub(r'\^(\w)|\^\{([^}]+)\}', rep_super, math_text)
    
    def rep_sub(m):
        content = m.group(1) or m.group(2)
        content = content.strip()
        converted = []
        for c in content:
            if c in sub_map:
                converted.append(sub_map[c])
            else:
                return f"_{content}"
        return "".join(converted)
        
    math_text = re.sub(r'\_(\w)|\_\{([^}]+)\}', rep_sub, math_text)
    
    # Clean up any leftover LaTeX commands
    math_text = re.sub(r'\\([a-zA-Z]+)', r'\1', math_text)
    return math_text

def process_all_math_blocks(text, to_unicode_math):
    def repl_block(m):
        content = m.group(1)
        if to_unicode_math:
            cleaned = process_math_content(content)
        else:
            cleaned = content
        return '$$' + cleaned + '$$'
        
    def repl_inline(m):
        content = m.group(1)
        if to_unicode_math:
            cleaned = process_math_content(content)
        else:
            cleaned = content
        return '$' + cleaned + '$'
        
    text = re.sub(r'\$\$(.*?)\$\$', repl_block, text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', repl_inline, text)
    return text

def get_clipboard_html():
    try:
        import ctypes
        from ctypes import wintypes
        
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        
        # Register format first (restype UINT, argtype LPCWSTR)
        user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterClipboardFormatW.restype = wintypes.UINT
        
        cf_html = user32.RegisterClipboardFormatW("HTML Format")
        if not cf_html:
            return None
            
        user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        
        if not user32.IsClipboardFormatAvailable(cf_html):
            return None
            
        # Open Clipboard (restype BOOL, argtype HWND)
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        
        if not user32.OpenClipboard(None):
            return None
            
        # GetClipboardData (restype c_void_p, argtype UINT)
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = ctypes.c_void_p
        
        h_data = user32.GetClipboardData(cf_html)
        if not h_data:
            user32.CloseClipboard()
            return None
            
        # GlobalLock (restype c_void_p, argtype c_void_p)
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        
        p_data = kernel32.GlobalLock(h_data)
        if not p_data:
            user32.CloseClipboard()
            return None
            
        # Read byte data safely using string_at
        html_bytes = ctypes.string_at(p_data)
        
        # GlobalUnlock (restype BOOL, argtype c_void_p)
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        kernel32.GlobalUnlock(h_data)
        
        # CloseClipboard
        user32.CloseClipboard.argtypes = []
        user32.CloseClipboard.restype = wintypes.BOOL
        user32.CloseClipboard()
        
        if html_bytes:
            return html_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading clipboard HTML: {e}")
    return None

def convert_html_to_markdown(html_data):
    start_idx = html_data.find("<!--StartFragment-->")
    end_idx = html_data.find("<!--EndFragment-->")
    if start_idx != -1 and end_idx != -1:
        html = html_data[start_idx + len("<!--StartFragment-->"):end_idx]
    else:
        html = html_data
        
    for _ in range(3): # Handle nesting up to 3 levels deep
        html = re.sub(r'<(?:b|strong)(?:\s[^>]*)?>(.*?)</(?:b|strong)>', r'**\1**', html, flags=re.DOTALL)
        html = re.sub(r'<(?:i|em)(?:\s[^>]*)?>(.*?)</(?:i|em)>', r'*\1*', html, flags=re.DOTALL)
        
        def parse_span_styles(m):
            attrs = m.group(1) or ""
            content = m.group(2)
            is_bold = 'font-weight' in attrs and any(x in attrs for x in ['bold', '700', '800', '900'])
            is_italic = 'font-style' in attrs and 'italic' in attrs
            if is_bold and is_italic:
                return f"***{content}***"
            elif is_bold:
                return f"**{content}**"
            elif is_italic:
                return f"*{content}*"
            return content
            
        html = re.sub(r'<span(?:\s+style="([^"]*)")?[^>]*>(.*?)</span>', parse_span_styles, html, flags=re.DOTALL)
        
    # Replace `<br>` and block elements to retain line breaks!
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(?:p|div|li|tr|h1|h2|h3|h4|h5|h6)>', '\n', html, flags=re.IGNORECASE)
    
    # Strip all other HTML tags
    text = re.sub(r'<[^>]+>', '', html)
    
    # Unescape HTML entities
    import html as html_module
    text = html_module.unescape(text)
    
    # Clean up excessive newlines while keeping structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

# Core text cleaning algorithm
def clean_pdf_text(text, unwrap=True, remove_hyphens=True, norm_spaces=True, fix_ligatures=True, clean_latex=True, to_unicode_math=False):
    if not text:
        return ""
        
    # 1. Normalize unicode (combines decomposed characters, fixing Vietnamese font issues)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Fix common PDF ligatures (which often render as weird single characters)
    if fix_ligatures:
        ligatures = {
            'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'ﬅ': 'ft', 'ﬆ': 'st', 'Æ': 'AE', 'æ': 'ae', 'Œ': 'OE', 'œ': 'oe'
        }
        for lig, repl in ligatures.items():
            text = text.replace(lig, repl)
            
    # 3. Handle line unwrapping (joining lines that belong to the same paragraph)
    if unwrap:
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split by double newlines (which indicate actual paragraph breaks in PDF text)
        paragraphs = re.split(r'\n\s*\n', text)
        cleaned_paragraphs = []
        
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue
                
            # Unwrap lines while preserving list bullets and numbering
            lines = para.split('\n')
            unwrapped_lines = []
            current_line = ""
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                # Check if it matches a bullet or numbered list pattern:
                # e.g., •, -, *, +, ⁃, ◦, ▪, ▫, 1., a., 1), a)
                is_bullet = re.match(r'^\s*(?:[•\-\*\+⁃◦▪▫\u2022\u2043\u25E6\u25AA\u25AB]|[a-zA-Z\d]+[\.\)])\s+', line)
                if is_bullet or not current_line:
                    if current_line:
                        unwrapped_lines.append(current_line.strip())
                    current_line = line
                else:
                    curr_stripped = current_line.rstrip()
                    if remove_hyphens and curr_stripped.endswith('-'):
                        current_line = curr_stripped[:-1] + line_str
                    else:
                        current_line = current_line + " " + line_str
            if current_line:
                unwrapped_lines.append(current_line.strip())
            
            if norm_spaces:
                unwrapped_lines = [re.sub(r'\s+', ' ', l).strip() for l in unwrapped_lines]
            para = '\n'.join(unwrapped_lines)
            
            if para:
                cleaned_paragraphs.append(para)
                
        text = '\n\n'.join(cleaned_paragraphs)
    else:
        # If not unwrapping, just clean up individual lines
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            if norm_spaces:
                line = re.sub(r'\s+', ' ', line)
            cleaned_lines.append(line.strip())
        text = '\n'.join(cleaned_lines)
        
    # 4. Handle LaTeX cleaning and formatting
    if clean_latex:
        text = clean_latex_formatting(text)
        text = wrap_math_in_delimiters(text)
        text = wrap_block_equations_logic(text)
        
        # Split text into math blocks and non-math blocks to protect math markup
        placeholders = []
        def store_placeholder(m):
            placeholders.append(m.group(0))
            return f"___MATH_BLOCK_PLACEHOLDER_{len(placeholders)-1}___"
            
        # Store $$...$$ and $...$
        text = re.sub(r'\$\$.*?\$\$', store_placeholder, text, flags=re.DOTALL)
        text = re.sub(r'\$.*?\$', store_placeholder, text)
        
        # Now apply bold/italic conversion only on the regular non-math text
        text = apply_bold_italic_formatting(text)
        text = convert_markdown_bold_italic(text)
        
        # Restore the math blocks
        for i, ph in enumerate(placeholders):
            text = text.replace(f"___MATH_BLOCK_PLACEHOLDER_{i}___", ph)
        
    # 5. Handle math block cleaning and/or Unicode math conversion
    if clean_latex or to_unicode_math:
        text = process_all_math_blocks(text, to_unicode_math)
        
    return text

class PDFTextCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF to Word Text Cleaner")
        self.root.geometry("720x420")
        self.root.minsize(580, 360)
        
        # Catppuccin Mocha themed colors for a premium dark look
        self.colors = {
            "bg": "#1e1e2e",
            "card": "#252538",
            "text": "#cdd6f4",
            "text_muted": "#a6adc8",
            "accent": "#89b4fa",
            "success": "#a6e3a1",
            "danger": "#f38ba8",
            "text_dark": "#11111b",
            "textbox_bg": "#313244",
            "border": "#45475a"
        }
        
        self.root.configure(bg=self.colors["bg"])
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure('TFrame', background=self.colors["bg"])
        
        self.setup_ui()
        
    def toggle_settings(self):
        if self.settings_visible:
            self.control_frame.pack_forget()
            self.btn_settings.config(text="⚙️ Hiện thiết lập", bg=self.colors["card"], fg=self.colors["accent"])
            self.settings_visible = False
            # Reduce window size smoothly
            current_w = self.root.winfo_width()
            current_h = self.root.winfo_height()
            self.root.geometry(f"{current_w}x{max(360, current_h - 110)}")
        else:
            self.settings_visible = True
            self.btn_settings.config(text="⚙️ Ẩn thiết lập", bg=self.colors["border"], fg=self.colors["text"])
            # Pack settings panel right after the input text box
            self.control_frame.pack(fill=tk.X, pady=(0, 8), padx=2, after=self.input_text)
            # Increase window size smoothly
            current_w = self.root.winfo_width()
            current_h = self.root.winfo_height()
            self.root.geometry(f"{current_w}x{current_h + 110}")

    def setup_ui(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.colors["bg"], padx=15, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header Frame
        header_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        header_frame.pack(fill=tk.X, pady=(0, 2))
        
        # Header Label
        header_label = tk.Label(
            header_frame, 
            text="PDF TEXT CLEANER & UNWRAPPER", 
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["accent"]
        )
        header_label.pack(side=tk.LEFT)
        
        # Toggle Settings Button
        self.settings_visible = False
        self.btn_settings = tk.Button(
            header_frame,
            text="⚙️ Hiện thiết lập",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["card"],
            fg=self.colors["accent"],
            activebackground=self.colors["border"],
            activeforeground=self.colors["accent"],
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self.toggle_settings
        )
        self.btn_settings.pack(side=tk.RIGHT)
        
        subheader_label = tk.Label(
            main_frame, 
            text="Tự động làm sạch khoảng trắng, nối đoạn bị xuống dòng lỗi và xóa dấu gạch nối từ PDF sang Word.",
            font=("Segoe UI", 9, "italic"),
            bg=self.colors["bg"],
            fg=self.colors["text_muted"]
        )
        subheader_label.pack(anchor=tk.W, pady=(0, 6))
        
        # Top half: Input Area
        input_label_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        input_label_frame.pack(fill=tk.X, anchor=tk.W, pady=(0, 3))
        
        tk.Label(
            input_label_frame,
            text="Nội dung gốc (Ctrl+V vào đây):",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT)
        
        # Input Text box
        self.input_text = tk.Text(
            main_frame,
            wrap=tk.WORD,
            bg=self.colors["textbox_bg"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],  # Cursor color
            font=("Segoe UI", 10),
            bd=1,
            relief=tk.FLAT,
            padx=8,
            pady=8,
            height=4
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.input_text.bind("<Control-a>", self.select_all)
        self.input_text.bind("<KeyRelease>", self.on_key_release)
        self.input_text.bind("<Control-v>", self.custom_paste)
        self.input_text.bind("<Shift-Insert>", self.custom_paste)
        self.input_text.bind("<Button-3>", self.show_context_menu)
        
        # Context Menu for right-click
        self.context_menu = tk.Menu(self.root, tearoff=0, bg=self.colors["card"], fg=self.colors["text"], activebackground=self.colors["accent"], activeforeground=self.colors["text_dark"], bd=0)
        self.context_menu.add_command(label="Cắt (Cut)", command=lambda: self.input_text.event_generate("<<Cut>>"))
        self.context_menu.add_command(label="Sao chép (Copy)", command=lambda: self.input_text.event_generate("<<Copy>>"))
        self.context_menu.add_command(label="Dán định dạng (Paste Rich)", command=self.custom_paste)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Xóa tất cả (Clear)", command=self.action_clear)
        
        # Control Panel Options & Action Buttons (Hidden by default, toggled via self.btn_settings)
        self.control_frame = tk.Frame(main_frame, bg=self.colors["card"], bd=1, relief=tk.SOLID, highlightbackground=self.colors["border"])
        self.control_frame.configure(highlightthickness=1, highlightcolor=self.colors["border"], bg=self.colors["card"])
        
        # Inner padding frame for control panel
        inner_control = tk.Frame(self.control_frame, bg=self.colors["card"], padx=15, pady=10)
        inner_control.pack(fill=tk.X)
        
        # Checkboxes variables
        self.var_unwrap = tk.BooleanVar(value=True)
        self.var_hyphens = tk.BooleanVar(value=True)
        self.var_spaces = tk.BooleanVar(value=True)
        self.var_ligatures = tk.BooleanVar(value=True)
        self.var_clean_latex = tk.BooleanVar(value=True)
        self.var_unicode_math = tk.BooleanVar(value=False)
        self.var_autoclean = tk.BooleanVar(value=True)
        
        # Checkbox widgets
        cb_unwrap = tk.Checkbutton(
            inner_control, text="Nối các dòng lỗi (Unwrap)", variable=self.var_unwrap,
            bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["text"],
            font=("Segoe UI", 9), command=self.trigger_clean
        )
        cb_unwrap.grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=2)
        
        cb_hyphens = tk.Checkbutton(
            inner_control, text="Xóa dấu nối cuối dòng (-)", variable=self.var_hyphens,
            bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["text"],
            font=("Segoe UI", 9), command=self.trigger_clean
        )
        cb_hyphens.grid(row=0, column=1, sticky=tk.W, padx=(0, 20), pady=2)
        
        cb_spaces = tk.Checkbutton(
            inner_control, text="Chuẩn hóa khoảng trắng", variable=self.var_spaces,
            bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["text"],
            font=("Segoe UI", 9), command=self.trigger_clean
        )
        cb_spaces.grid(row=0, column=2, sticky=tk.W, padx=(0, 20), pady=2)
        
        cb_ligatures = tk.Checkbutton(
            inner_control, text="Sửa lỗi Font / Ligature", variable=self.var_ligatures,
            bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["text"],
            font=("Segoe UI", 9), command=self.trigger_clean
        )
        cb_ligatures.grid(row=1, column=0, sticky=tk.W, padx=(0, 20), pady=2)
        
        cb_clean_latex = tk.Checkbutton(
            inner_control, text="Chuẩn hóa LaTeX ($...$)", variable=self.var_clean_latex,
            bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["text"],
            font=("Segoe UI", 9), command=self.trigger_clean
        )
        cb_clean_latex.grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=2)
        
        cb_unicode_math = tk.Checkbutton(
            inner_control, text="Chuyển Unicode Math (α, ², ₜ)", variable=self.var_unicode_math,
            bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["text"],
            font=("Segoe UI", 9), command=self.trigger_clean
        )
        cb_unicode_math.grid(row=1, column=2, sticky=tk.W, padx=(0, 20), pady=2)
        
        cb_autoclean = tk.Checkbutton(
            inner_control, text="Tự động xử lý & Copy khi dán/nhập", variable=self.var_autoclean,
            bg=self.colors["card"], fg=self.colors["accent"], selectcolor=self.colors["bg"],
            activebackground=self.colors["card"], activeforeground=self.colors["accent"],
            font=("Segoe UI", 9, "bold")
        )
        cb_autoclean.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Action Buttons frame
        btn_frame = tk.Frame(inner_control, bg=self.colors["card"])
        btn_frame.grid(row=0, column=3, rowspan=3, sticky=tk.NSEW, padx=(20, 0))
        inner_control.grid_columnconfigure(3, weight=1)
        
        # Custom styled buttons
        self.btn_clean = tk.Button(
            btn_frame,
            text="LÀM SẠCH & COPY",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["success"],
            fg=self.colors["text_dark"],
            activebackground=self.colors["accent"],
            activeforeground=self.colors["text_dark"],
            bd=0,
            padx=12,
            pady=5,
            cursor="hand2",
            command=self.action_clean_and_copy
        )
        self.btn_clean.pack(side=tk.RIGHT, padx=4)
        
        self.btn_clear = tk.Button(
            btn_frame,
            text="XÓA HẾT",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["danger"],
            fg=self.colors["text_dark"],
            activebackground=self.colors["text_muted"],
            activeforeground=self.colors["text_dark"],
            bd=0,
            padx=10,
            pady=5,
            cursor="hand2",
            command=self.action_clear
        )
        self.btn_clear.pack(side=tk.RIGHT, padx=4)
        
        # Bottom half: Output Area
        tk.Label(
            main_frame,
            text="Kết quả đã xử lý (Tự động cập nhật):",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["success"]
        ).pack(anchor=tk.W, pady=(0, 3))
        
        # Output Text box
        self.output_text = tk.Text(
            main_frame,
            wrap=tk.WORD,
            bg=self.colors["textbox_bg"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            font=("Segoe UI", 10),
            bd=1,
            relief=tk.FLAT,
            padx=8,
            pady=8,
            height=4
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.output_text.bind("<Control-a>", self.select_all)
        
        # Status Bar
        self.status_label = tk.Label(
            main_frame,
            text="Sẵn sàng xử lý dữ liệu.",
            font=("Segoe UI", 9),
            bg=self.colors["bg"],
            fg=self.colors["text_muted"]
        )
        self.status_label.pack(side=tk.LEFT, pady=2)
        
        # Instructions Label
        instructions = tk.Label(
            main_frame,
            text="Mẹo: Nhấn Ctrl+V vào ô trên để tự động dọn dẹp và copy kết quả.",
            font=("Segoe UI", 8, "italic"),
            bg=self.colors["bg"],
            fg=self.colors["text_muted"]
        )
        instructions.pack(side=tk.RIGHT, pady=2)

    def select_all(self, event):
        event.widget.tag_add("sel", "1.0", "end")
        return "break"

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def custom_paste(self, event=None):
        html_data = get_clipboard_html()
        if html_data:
            markdown_text = convert_html_to_markdown(html_data)
            if markdown_text:
                try:
                    self.input_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
                except tk.TclError:
                    pass
                self.input_text.insert(tk.INSERT, markdown_text)
                
                # Automatically process and clean the pasted markdown text
                if self.var_autoclean.get():
                    cleaned = self.trigger_clean()
                    if cleaned:
                        self.copy_to_clipboard(cleaned)
                        self.show_status("Đã tự động làm sạch & copy vào Clipboard!", self.colors["success"])
                return "break"
        return None

    def trigger_clean(self):
        # Read text from input
        raw_text = self.input_text.get("1.0", tk.END).strip()
        if not raw_text:
            self.output_text.delete("1.0", tk.END)
            return ""
            
        cleaned_with_delimiters = clean_pdf_text(
            raw_text,
            unwrap=self.var_unwrap.get(),
            remove_hyphens=self.var_hyphens.get(),
            norm_spaces=self.var_spaces.get(),
            fix_ligatures=self.var_ligatures.get(),
            clean_latex=self.var_clean_latex.get(),
            to_unicode_math=self.var_unicode_math.get()
        )
        
        # For display, strip delimiters so it shows as clean plain text
        cleaned_display = re.sub(r'\$\$(.*?)\$\$', r'\1', cleaned_with_delimiters, flags=re.DOTALL)
        cleaned_display = re.sub(r'\$(.*?)\$', r'\1', cleaned_display)
        
        # Update output
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", cleaned_display)
        return cleaned_with_delimiters

    def on_key_release(self, event):
        # On key release, if autoclean is checked, clean and update
        if self.var_autoclean.get():
            cleaned = self.trigger_clean()
            # If paste event or modified significantly, copy to clipboard
            # (In Tkinter, Ctrl+v paste triggers KeyRelease. We check if text is cleaned successfully)
            if cleaned and event.keysym in ('v', 'V', 'Control_L', 'Control_R'):
                self.copy_to_clipboard(cleaned)
                self.show_status("Đã tự động làm sạch & copy vào Clipboard!", self.colors["success"])

    def action_clean_and_copy(self):
        cleaned = self.trigger_clean()
        if cleaned:
            self.copy_to_clipboard(cleaned)
            self.show_status("Đã làm sạch và copy kết quả thành công!", self.colors["success"])
        else:
            self.show_status("Không có nội dung để xử lý.", self.colors["danger"])

    def action_clear(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.input_text.focus_set()
        self.show_status("Đã xóa toàn bộ nội dung.", self.colors["text_muted"])

    def copy_to_clipboard(self, text):
        # Plain text payload should have delimiters stripped
        plain_text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
        plain_text = re.sub(r'\$(.*?)\$', r'\1', plain_text)
        
        # HTML version wraps formulas in Cambria Math via markdown_to_html
        html_version = markdown_to_html(text)
        
        success = set_clipboard_html_and_text(html_version, plain_text)
        if not success:
            self.root.clipboard_clear()
            self.root.clipboard_append(plain_text)
            self.root.update()

    def show_status(self, message, color):
        self.status_label.config(text=message, fg=color)
        # Clear status message after 3 seconds back to default
        self.root.after(3000, lambda: self.status_label.config(text="Sẵn sàng xử lý dữ liệu.", fg=self.colors["text_muted"]))

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFTextCleanerApp(root)
    root.mainloop()

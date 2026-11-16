
from app.services.chunking import chunk_text


sample_text = """
Plagiarism – how do I avoid it?

The following good practice guidelines will help you avoid plagiarism. Students should understand the importance of academic integrity and learn how to acknowledge the work of others correctly.

Use of Quotations

Remember that exact words from your source should appear in quotation marks and be referenced. Always include the appropriate page number and full reference. Use direct quotations sparingly and only when necessary.

Making Notes

During note taking, it is possible to use the language of your source subconsciously. Read the material first, understand the author's ideas, and then write notes in your own words. This helps prevent accidental copying.

Paraphrasing

Attribute the broad ideas or content to the original author. Explain the ideas in your own words while making it clear which source you used. Changing only a few words is not sufficient paraphrasing.

Cite all sources used.

Always cite the sources that contributed to your assignment. This includes books, academic articles, websites, and other materials. When in doubt, ask your tutor for guidance.
"""


chunks = chunk_text(sample_text)

for index, chunk in enumerate(chunks):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
    print("Characters:", len(chunk))

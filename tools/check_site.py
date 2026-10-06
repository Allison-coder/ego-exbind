"""Local browser smoke tests. Requires Playwright and a Chromium executable."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification"
OUT.mkdir(exist_ok=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium-browser",
                                    headless=False, args=["--headless=new", "--no-sandbox", "--disable-dev-shm-usage"])
        report = []
        for width, height in [(1440, 1000), (390, 844), (320, 740)]:
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1,
                                    reduced_motion="reduce")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((ROOT / "index.html").as_uri())
            # Load lazy images before recording layout and asset integrity.
            page.evaluate("document.querySelectorAll('img').forEach(i => i.loading = 'eager')")
            page.wait_for_function("Array.from(document.querySelectorAll('img[src]')).every(i => i.complete && i.naturalWidth > 0)")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal page overflow"
            assert page.locator(".actions a").first.get_attribute("href") == "https://github.com/Allison-coder/ego-exbind"
            assert page.locator(".actions a").count() == 1
            assert page.locator(".author-list").inner_text() == "Yuqi Guo1, Michael Wray1,*"
            assert page.locator(".author-list a").get_attribute("href") == "https://mwray.github.io/"
            assert page.locator(".affiliation").inner_text() == "1University of Bristol"
            assert page.locator(".corresponding-author").inner_text() == "*Corresponding author"
            assert page.locator(".adaptation-table tbody tr").last.locator("td").all_text_contents() == ["32.02", "0.055", "0.077"]
            assert "means over three random seeds" in page.locator("#interventions .source-note").inner_text()
            assert "Dissertation Fig. 2.1" not in page.locator("body").inner_text()
            assert "Page design inspired by" not in page.locator("footer").inner_text()
            assert "Original PDF" not in page.locator("body").text_content()
            for reference in ["thesis", "dissertation", "Table 4.", "Tables 4."]:
                assert reference not in page.locator("body").text_content()
            assert page.locator("#panel-retrieval figcaption a").count() == 0
            assert page.locator("#panel-retrieval .figure-surface > img").get_attribute("src").endswith("retrieval.svg")
            assert page.locator(".research-summary").count() == 1
            assert page.locator(".action-strip img").get_attribute("src").endswith("actions.webp")
            assert "EgoVLPv2, pretrained on EgoClip" in page.locator("#setup").inner_text()
            assert page.locator(".contribution-index li").count() == 3
            assert page.locator(".flow-branches > div").count() == 2
            assert "Similarity anchoring" in page.locator("#interventions").inner_text()
            assert "Noun-margin protection" in page.locator("#interventions").inner_text()
            assert "33.33" in page.locator("#interventions").inner_text()
            assert page.locator(".evaluation-flow li").count() == 4
            assert page.locator("#overview").evaluate("el => el.compareDocumentPosition(document.querySelector('#setup')) & Node.DOCUMENT_POSITION_FOLLOWING")
            for link in page.locator('.nav-links a[href^="#"]').all():
                assert page.locator(link.get_attribute("href")).count() == 1
            page.locator('[data-figure="retrieval"]').click()
            assert page.locator("#dialog-image").get_attribute("src").endswith("retrieval.svg")
            page.locator("#close-dialog").click()
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.screenshot(path=str(OUT / (str(width) + "-home.png")))
            page.screenshot(path=str(OUT / (str(width) + "-full.png")), full_page=True)
            page.evaluate("window.scrollTo(0, document.querySelector('#interventions').offsetTop - document.querySelector('.site-header').offsetHeight)")
            page.wait_for_timeout(300)
            assert abs(page.locator('.site-header').bounding_box()['y']) < 1
            page.screenshot(path=str(OUT / (str(width) + "-interventions.png")))
            page.locator("#tab-pmi").click()
            assert page.locator("#panel-pmi").is_visible()
            assert not page.locator("#panel-retrieval").is_visible()
            page.locator('[data-figure="pmi"]').click()
            assert page.locator("#figure-dialog").is_visible()
            page.keyboard.press("Escape")
            assert not page.locator("#figure-dialog").is_visible()
            page.locator("#tab-retrieval").click()
            page.locator("#tab-splits").click()
            assert page.locator("#panel-splits").is_visible()
            assert "24.51" in page.locator("#panel-splits").inner_text()
            page.locator("#tab-splits").press("ArrowRight")
            assert page.locator("#panel-margins").is_visible()
            assert "0.065" in page.locator("#panel-margins").inner_text()
            page.locator("#tab-overall").click()
            assert "33.70" in page.locator("#panel-overall").inner_text()
            page.locator("#results").scroll_into_view_if_needed()
            page.screenshot(path=str(OUT / (str(width) + "-results.png")))
            assert not errors, errors
            report.append({"viewport": [width, height], "assets": "loaded", "tabs": "passed",
                           "dialog": "passed", "overflow": False, "js_errors": errors})
            page.close()
        browser.close()
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

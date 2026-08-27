---
title: Strategie für automatisierte Tests
description: Automatische Durchführung und regelmäßige Ausführung von
  Software-Tests.
category:
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/automated-tests/
problems:
- legacy-code-without-tests
- poor-test-coverage
- test-debt
- inadequate-integration-tests
- missing-end-to-end-tests
- flaky-tests
- difficult-to-test-code
- regression-bugs
- outdated-tests
- testing-complexity
- testing-environment-fragility
- inadequate-test-data-management
- inadequate-test-infrastructure
- increased-manual-testing-effort
- high-bug-introduction-rate
- high-defect-rate-in-production
- insufficient-testing
layout: solution
lang: de
en_slug: test-coverage-strategy
related_solutions:
- slug: automated-tests
  similarity: 0.9
- slug: regression-testing
  similarity: 0.85
- slug: functional-tests
  similarity: 0.85
- slug: characterization-tests
  similarity: 0.85
- slug: code-coverage-analysis
  similarity: 0.8
- slug: ci-cd-pipeline
  similarity: 0.8
---

## Description

Automatisierte Tests, regelmäßig statt manuell ausgeführt, verifizieren, dass sich Software nach einer Änderung noch wie erwartet verhält — aber diese Abdeckung in ein Legacy-System einzubauen, das keine hat, bedeutet, mit Charakterisierungstests zu beginnen, die aufzeichnen, was der Code heute tatsächlich tut, einschließlich seiner Fehler, statt direkt zu behaupten, was er tun sollte. Da eine Legacy-Codebasis ohne Tests jede Änderung zu einem Glücksspiel gegen stille Regression macht, zielt die richtige Strategie auf Abdeckung dort, wo sie die meiste Sicherheit pro investierter Stunde liefert: die Zahlungsflüsse, die Authentifizierungslogik und die Integrationspunkte, die sich häufig ändern und echten Schaden verursachen, wenn sie brechen, statt einheitlicher Abdeckung über Code hinterherzujagen, den niemand anfasst. Bedeutsame Abdeckung auf diese Weise zu erreichen braucht Monate konsistenten, disziplinierten Aufwands, und ein schlecht geschriebener Test — einer, der etwas so Breites behauptet, dass er nie fehlschlagen kann — erzeugt genau das falsche Vertrauen, das eine echte Testsuite verhindern soll.

## How to Apply ◆

> Legacy-Systeme operieren typischerweise ohne automatisierte Tests, was jede Änderung zu einem Glücksspiel gegen Regressionen macht; der strategische Aufbau von Testabdeckung — beginnend mit dem risikoreichsten Code statt einheitlicher Abdeckung nachzujagen — liefert Sicherheit dort, wo sie am meisten zählt, ohne die Lieferung zu stoppen.

- Beginnen Sie mit dem Schreiben von Charakterisierungstests, nicht Korrektheitstests. In Legacy-Systemen ist das Ziel zunächst, zu erfassen, was der Code tatsächlich tut, einschließlich seiner Fehler und undokumentierten Verhaltensweisen, sodass strukturelle Änderungen sicher vorgenommen werden können. Charakterisierungstests protokollieren das aktuelle Verhalten; Korrektheit kommt später.
- Priorisieren Sie Testabdeckung für den Code, der sich am häufigsten ändert und den meisten Schaden verursacht, wenn er bricht: Zahlungsflüsse, Authentifizierungslogik, Datentransformationspipelines und Integrationspunkte mit externen Systemen. Diese Bereiche gründlich abzudecken liefert weit mehr Wert als einheitliche Abdeckung über die gesamte Codebasis zu erreichen.
- Nutzen Sie die Testpyramide bewusst in Legacy-Kontexten: Investieren Sie stark in Unit-Tests für isolierte Geschäftslogik, wo Nähte eingeführt werden können, nutzen Sie Integrationstests, um Datenbankabfragen, API-Aufrufe und Komponenteninteraktionen zu verifizieren, und beschränken Sie End-to-End-Tests auf die kritischsten Nutzerreisen, wo das Gesamtsystemverhalten bestätigt werden muss.
- Wenden Sie Michael Feathers' Naht-Techniken an, um ungetesteten Legacy-Code testbar zu machen, ohne ihn neu zu schreiben: Führen Sie Schnittstellen an Abhängigkeitsgrenzen ein, nutzen Sie Dependency Injection, um hartcodierte Mitarbeiter durch Test-Doubles zu ersetzen, und extrahieren Sie Logik aus framework-gekoppelten Klassen in einfache Objekte, die isoliert getestet werden können.
- Setzen Sie eine No-Regression-Richtlinie durch: Jede Fehlerbehebung muss von einem Test begleitet werden, der den Fehler erfasst hätte. Diese Praxis baut Abdeckung genau dort auf, wo das System historisch brüchig war, und stellt sicher, dass Defekte nicht wiederholt erneut eingeführt werden.
- Verfolgen Sie Codeabdeckung als Untergrenze, nicht als Ziel. Das Setzen einer Abdeckungsschwelle (zum Beispiel 60 %), die die Pipeline durchsetzt, verhindert, dass die Abdeckung sinkt, während neuer Code hinzugefügt wird, während gleichzeitig die Falle vermieden wird, bedeutungslose Tests zu schreiben, nur um den Prozentsatz aufzublähen.
- Isolieren Sie flaky Tests sofort und untersuchen Sie sie als Defekte. Legacy-Systeme haben oft nichtdeterministisches Verhalten, verursacht durch globalen Zustand, Timing-Abhängigkeiten oder inkonsistente Testdaten; ein flaky Test, den das Team ignorieren lernt, zerstört das Vertrauen in die gesamte Suite.
- Erstellen und pflegen Sie eine dedizierte Testdatenstrategie: Vermeiden Sie gemeinsam genutzte veränderliche Testdaten, die dazu führen, dass sich Tests gegenseitig stören, und bauen Sie Factory-Funktionen oder Builder-Muster, die isolierte, selbstbeschreibende Testdatensätze erstellen. Legacy-Datenbanken mit produktionsabgeleiteten Testdaten sind eine häufige Quelle von Test-Brüchigkeit.

## Tradeoffs ⇄

> Automatisierte Testabdeckung verwandelt ein Legacy-System von einem brüchigen Artefakt, das nicht sicher geändert werden kann, in eine Codebasis, die Entwickler mit Zuversicht refaktorieren, erweitern und deployen können.

**Vorteile:**

- Bietet das Sicherheitsnetz, das Refactoring in einer Legacy-Codebasis möglich macht: Ohne Tests ist die Verbesserung der Codestruktur ein Glücksspiel; mit Tests kann jede Transformation als verhaltenserhaltend verifiziert werden.
- Erfasst Regressionen unmittelbar nach der Änderung, die sie verursacht hat, wenn der Entwickler noch vollen Kontext hat, statt Wochen später in einem manuellen Testzyklus.
- Reduziert die manuelle Testlast bei jedem Release und setzt QA-Zeit für explorative Tests neuen Verhaltens frei statt für Re-Verifikation bestehender Funktionalität, die Automatisierung handhaben kann.
- Dient als ausführbare Dokumentation des tatsächlichen Verhaltens des Legacy-Systems und ergänzt oder ersetzt schriftliche Spezifikationen, die über Jahre aus der Synchronisation mit dem Code geraten sind.
- Ermöglicht der CI/CD-Pipeline, bedeutsames Deployment-Vertrauen zu liefern, und verwandelt automatisierte Tests in das primäre Qualitätsgate statt in eine ergänzende Prüfung.

**Kosten und Risiken:**

- Legacy-Code, der nicht für Testbarkeit konzipiert wurde — mit globalem Zustand, hartcodierten Abhängigkeiten und framework-gekoppelter Logik —, erfordert erhebliche Umstrukturierung, bevor Unit-Tests geschrieben werden können, was die anfängliche Investition höher macht als bei Greenfield-Systemen.
- Charakterisierungstests, die aktuelle Fehler und falsches Verhalten erfassen, müssen sorgfältig verwaltet werden: Sie sind während struktureller Refaktorierung nützlich, müssen aber aktualisiert oder ersetzt werden, wenn diese Fehler eventuell behoben werden, sonst zementieren sie falsches Verhalten dauerhaft.
- Bedeutsame Testabdeckung auf einer großen Legacy-Codebasis zu erreichen braucht Monate oder Jahre konsistenten Aufwands; Teams, die schnelle Ergebnisse erwarten, könnten die Praxis aufgeben, bevor das Sicherheitsnetz stark genug ist, um Verhalten zu ändern.
- Schlecht geschriebene Tests — die Implementierungsdetails statt Verhalten testen oder so breite Behauptungen aufstellen, dass sie nie fehlschlagen — erzeugen falsches Vertrauen, während sie Wartungs-Overhead hinzufügen, ohne echten Schutz zu bieten.
- Langsame Testsuiten, die Stunden zum Ausführen brauchen, sind in Legacy-Systemen üblich, wo sich Integrationstests gegen echte Datenbanken und externe Systeme ansammeln; ohne aktive Verwaltung wird die Pipeline zu einem Engpass, um den Teams herumarbeiten statt durch ihn hindurch.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie eine bewusste Testabdeckungsstrategie ein Legacy-System unter Kontrolle bringt, ohne eine störende, die-Welt-anhaltende Testanstrengung zu erfordern.

Ein Logistik-Softwareunternehmen erbte ein Legacy-Routenoptimierungssystem aus einer Übernahme. Das System hatte null automatisierte Tests und eine Historie von Regressionen bei jedem neuen Release, was die neuen Eigentümer dazu brachte, alle Feature-Entwicklung zu stoppen, während jede Änderung manuell verifiziert wurde. Statt umfassende Abdeckung zu versuchen, verbrachte das Team ihren ersten Monat damit, die fünf am häufigsten kaputten Verhaltensweisen in den vorangegangenen zwölf Monaten Fehlerberichten zu identifizieren: Mautberechnung, Gewichtsgrenzendurchsetzung, Gefahrgut-Routing-Regeln, Mehrfachstopp-Sequenzierung und Zeitfenster-Validierung. Sie schrieben Charakterisierungstests für jeden dieser fünf Bereiche, die die in Produktionsprotokollen beobachteten Eingaben und Ausgaben abdeckten. Innerhalb von sechs Wochen hatten sie 340 Tests, die die risikoreichsten Verhaltensweisen abdeckten. Regressionen in diesen Bereichen sanken innerhalb des Quartals auf null, und das Team nahm die Feature-Entwicklung mit einem gezielten Sicherheitsnetz wieder auf.

Eine europäische Bank, die ein Legacy-Handelsabwicklungssystem betrieb, hatte eine Testsuite von 4.000 End-to-End-Tests, die acht Stunden zum Ausführen brauchte. Die Suite war nominell umfassend, aber so langsam und brüchig, dass das Team sie nur einmal pro Woche laufen ließ und einzelne Fehlschläge ignorierte, es sei denn, sie wiederholten sich. Das Sicherheitsnetz existierte auf dem Papier, bot aber keinen praktischen Schutz. Ein neu eingestellter Testarchitekt analysierte die Suite und fand, dass 90 % der Szenarien mehrfach durch verschiedene End-to-End-Tests abgedeckt wurden. Das Team verbrachte drei Monate damit, redundante End-to-End-Tests durch fokussierte Integrationstests und gezielte Unit-Tests für die Geschäftslogik zu ersetzen, die diese End-to-End-Tests verifiziert hatten. Die neue Suite von 1.200 Tests lief in unter zwanzig Minuten. Mit der nun schnell genug für jeden Pull Request laufenden Pipeline entdeckte und behob das Team im ersten Monat drei Regressionen, die der vorherige wöchentliche Lauf zu spät erfasst hätte, um sie ihrer verursachenden Änderung zuzuordnen.

Ein Versicherungsunternehmen, das ein Legacy-Schadensverarbeitungssystem pflegte, stand vor einer kritischen Modernisierung: dem Ersatz der zugrunde liegenden Regel-Engine durch eine neuere Plattform. Die Schadenslogik war nie automatisch getestet worden, und das Team war von der Angst gelähmt, dass der Ersatz die Berechnung von Schadenssummen still verändern würde. Bevor sie irgendeinen Produktionscode anfassten, verbrachte das Team zwei Monate damit, eine Charakterisierungstestsuite zu bauen, indem sie Tausende historischer Schäden durch das bestehende System laufen ließen und die Ausgaben protokollierten. Diese Ausgaben wurden zu den erwarteten Werten für dieselben Schäden, ausgeführt durch die neue Regel-Engine. Die Charakterisierungstests identifizierten während des Paralleltests 23 Berechnungsunterschiede zwischen der alten und der neuen Engine, von denen 19 echte Fehler in der Legacy-Engine waren, die in der neuen Implementierung korrigiert wurden, und 4 beabsichtigte Verhaltensunterschiede, die geschäftliche Freigabe erforderten. Ohne die Charakterisierungstestsuite wäre die Migration unmöglich zu validieren gewesen; mit ihr wurde die Umschaltung mit dokumentiertem Vertrauen durchgeführt.

---
title: Refactoring
description: Regelmäßige Umstrukturierung von Code, ohne das externe Verhalten zu
  ändern.
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/refactoring/
problems:
- spaghetti-code
- god-object-anti-pattern
- high-coupling-low-cohesion
- bloated-class
- excessive-class-size
- circular-dependency-problems
- code-duplication
- copy-paste-programming
- tight-coupling-issues
- monolithic-functions-and-classes
- refactoring-avoidance
- feature-creep-without-refactoring
- workaround-culture
- accumulation-of-workarounds
- increasing-brittleness
- clever-code
- complex-and-obscure-logic
- poor-encapsulation
- tangled-cross-cutting-concerns
- over-reliance-on-utility-classes
- global-state-and-side-effects
- hardcoded-values
- large-pull-requests
- strangler-fig-pattern-failures
- system-stagnation
- circular-references
- maintenance-cost-increase
- technical-architecture-limitations
layout: solution
lang: de
en_slug: incremental-refactoring
related_solutions:
- slug: preparatory-refactoring
  similarity: 0.8
- slug: code-review-process-reform
  similarity: 0.8
- slug: refactoring-katas
  similarity: 0.8
- slug: code-metrics
  similarity: 0.8
- slug: static-analysis-and-linting
  similarity: 0.8
- slug: clean-code
  similarity: 0.75
---

## Description

Refactoring strukturiert die interne Form von Code um — eine Methode extrahieren, eine Gottklasse aufbrechen, eine Bedingung durch Polymorphie ersetzen —, während sein externes Verhalten bewusst bewahrt wird, sodass der Code leichter zu bearbeiten wird, ohne zu ändern, was er tut. In Legacy-Systemen macht das angesammelte Ausmaß strukturellen Verfalls eine einmalige Bereinigung sowohl unpraktikabel als auch riskant, weshalb die Praxis kontinuierlich und proportional zur tatsächlichen Arbeit werden muss: die Pfadfinderregel, jeglichen Code, den ein Entwickler ohnehin anfasst, etwas sauberer zu hinterlassen, als er ihn vorgefunden hat, gestützt durch Characterization Tests, die das bestehende Verhalten — einschließlich seiner Eigenheiten — erfassen, bevor eine strukturelle Änderung versucht wird. So angewendet, zahlt sich Refactoring direkt in verringerter Zeit und verringertem Risiko für jede nachfolgende Änderung an diesem Code aus, obwohl der Versuch ohne angemessene Testabdeckung echt riskiert, einen undokumentierten Nebeneffekt zu verändern, von dem sich still ein anderer Teil des Systems abhängig gemacht hat.

## How to Apply ◆

> In Legacy-Systemen muss Refactoring als disziplinierte, kontinuierliche Praxis eingeführt werden statt als einmaliges Bereinigungsprojekt, weil das Ausmaß angesammelter Schulden jeden Versuch, alles auf einmal anzugehen, sowohl unpraktikabel als auch riskant macht.

- Wenden Sie die Pfadfinderregel konsequent an: Wann immer ein Entwickler an einem Legacy-Modul arbeitet, um einen Bug zu beheben oder ein Feature hinzuzufügen, hinterlässt er diesen spezifischen Code inkrementell sauberer, als er ihn vorgefunden hat, ohne unzusammenhängende Bereiche anzufassen. Dies hält den Refactoring-Aufwand proportional zur tatsächlichen Arbeit statt zu separaten Kosten.
- Schreiben Sie vor dem Refactoring jeglichen Legacy-Codes Characterization Tests, die das bestehende Verhalten erfassen — einschließlich Bugs und undokumentierter Eigenheiten. Diese Tests sind keine Aussagen über Korrektheit; sie erfassen, was der Code tatsächlich tut, damit strukturelle Änderungen als verhaltenserhaltend verifiziert werden können.
- Nutzen Sie benannte Refactoring-Operationen — Methode extrahieren, Methode verschieben, Bedingung durch Polymorphie ersetzen — statt Ad-hoc-Bearbeitung. Benannte Operationen haben definierte Mechaniken, die das Risiko verringern, Verhalten in ungetesteten Codepfaden versehentlich zu ändern.
- Committen Sie jeden Refactoring-Schritt separat von Feature- oder Bugfix-Änderungen. In Legacy-Codebasen ist diese Disziplin besonders wichtig: Strukturelle Änderungen mit Verhaltensänderungen zu mischen macht es unmöglich zu identifizieren, welche Änderung eine Regression eingeführt hat.
- Fokussieren Sie den Refactoring-Aufwand auf Code, der aktiv modifiziert wird. Das tief verworrene Modul, das seit zwei Jahren niemand angefasst hat, birgt Risiko, wenn es ohne zwingenden Grund refaktoriert wird; die Zahlungsverarbeitungsklasse, die jeden Monat neue Anforderungen erhält, ist hochwertig und hochpriorisiert.
- Nutzen Sie automatisierte IDE-Refactoring-Werkzeuge, wo immer möglich. In Legacy-Codebasen mit schwachen Typsystemen, schlecht benannten Bezeichnern oder ohne Testabdeckung sind automatisierte Umbenennungs- und Extraktionsoperationen sicherer als manuelle Bearbeitung, weil das Werkzeug alle Referenzen verfolgt.
- Identifizieren und adressieren Sie zuerst die gefährlichsten Code Smells: Gottklassen, die unzusammenhängende Verantwortlichkeiten ansammeln, tief verschachtelte bedingte Logik und Klassen mit Hunderten von Zeilen duplizierten Codes schaden der Wartbarkeit am meisten und sollten zerlegt werden, bevor neue Funktionalität hinzugefügt wird.
- Verfolgen Sie den Refactoring-Aufwand separat in der Sprint-Planung und kommunizieren Sie seinen Wert an Stakeholder in Begriffen der Liefergeschwindigkeit: „Dieses Refactoring wird die Zeit zum Hinzufügen neuer Zahlungsmethoden von zwei Wochen auf drei Tage verringern" ist ein konkreter Geschäftsfall, der die Investition rechtfertigt.

## Tradeoffs ⇄

> Inkrementelles Refactoring ist der einzige nachhaltige Ansatz, um die Qualität von Legacy-Code über die Zeit zu verbessern, erfordert aber konsequente Teamdisziplin und ein verlässliches Test-Sicherheitsnetz, um keine neuen Defekte einzuführen.

**Vorteile:**

- Verringert die Kosten und das Risiko künftiger Änderungen, indem die Codestruktur inkrementell vereinfacht wird, was jede nachfolgende Modifikation am Legacy-System schneller und sicherer macht.
- Verhindert die Anhäufung neuer Schichten technischer Schulden, indem der aktiv bearbeitete Code kontinuierlich verbessert wird, statt Verfall sich aufsummieren zu lassen.
- Baut echtes Teamverständnis der Legacy-Codebasis auf: Entwickler, die ein Modul refaktorieren, lernen dessen Struktur weit tiefer als solche, die es nur lesen, was die kollektive Fähigkeit des Teams verbessert, es zu pflegen.
- Schafft natürliche Gelegenheiten, Testabdeckung hinzuzufügen, während verifiziert wird, dass das Refactoring das Verhalten bewahrt hat, und baut so schrittweise das Sicherheitsnetz auf, das der Legacy-Codebasis fehlt.
- Vermeidet das hohe Risiko und die organisatorische Störung von Big-Bang-Neufassungen, indem kontinuierliche strukturelle Verbesserung geliefert wird, während das System betriebsfähig bleibt und die Feature-Lieferung weiterläuft.

**Kosten und Risiken:**

- Legacy-Code ohne angemessene Testabdeckung zu refaktorieren ist echt gefährlich: eine verhaltenserhaltend erscheinende Transformation kann einen undokumentierten Nebeneffekt verändern, von dem andere Teile des Systems abhängen.
- Legacy-Codebasen enthalten oft tief verwobene Module, in denen selbst kleine strukturelle Änderungen Modifikationen über viele Dateien hinweg erfordern, was das Merge-Konflikt-Risiko erhöht, wenn mehrere Entwickler aktiv sind.
- Teams unter ständigem Druck, neue Features zu liefern und Produktionsdefekte in Legacy-Systemen zu beheben, schützen Refactoring-Zeit selten, was dazu führt, dass die Praxis ohne explizite Management-Unterstützung schnell verblasst.
- Schlecht ausgeführtes Refactoring — gleichzeitiges Ändern von Struktur und Verhalten oder zu große, nicht sicher rückgängig zu machende Schritte — kann Legacy-Code schwerer verständlich und pflegbar machen als zuvor.
- Stakeholder, die Fortschritt an sichtbaren Features messen, könnten Refactoring als nicht produktiv wahrnehmen, was Reibung erzeugt, wenn Entwickler für strukturelle Verbesserungen an einem System eintreten, das „bereits funktioniert".

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie inkrementelles Refactoring den strukturellen Verfall in echten Legacy-Systemen angeht, ohne eine disruptive Generalüberholung zu erfordern.

Ein Gesundheitssoftwareunternehmen pflegte ein Patiententerminsystem, in dem die Kernbuchungslogik über fünfzehn Jahre inkrementeller Feature-Ergänzungen zu einer einzigen 2.400-Zeilen-Klasse angewachsen war. Jede neue Terminierungsregel wurde als weiterer Zweig in einer bereits tief verschachtelten bedingten Struktur hinzugefügt. Das Team führte eine Richtlinie ein, eine Methode pro bearbeitetem Ticket zu refaktorieren: Wann immer ein Entwickler die Klasse aus irgendeinem Grund anfasste, extrahierte er die von ihm modifizierte Methode in eine klar benannte Funktion und committete diese Extraktion separat, bevor er die funktionale Änderung vornahm. Nach acht Monaten konsequenter Anwendung war die Klasse in acht fokussierte, zusammenarbeitende Klassen zerlegt worden, was die durchschnittliche Zeit zur Implementierung neuer Terminierungsregeln von vier Tagen auf einen halben Tag verringerte.

Das Bestandsverwaltungssystem eines Fertigungsunternehmens enthielt umfangreiche Copy-Paste-Duplikation über seine Bestandsberechnungsroutinen hinweg: dieselbe Rundungs- und Währungsumrechnungslogik erschien in über vierzig separaten Methoden, jede über Jahre unabhängiger Modifikation leicht von den anderen abgewichen. Das Team verbrachte zwei Tage damit, Characterization Tests zu schreiben, die die Ausgabe jeder Methode über eine repräsentative Menge von Eingaben erfassten. Mit diesen Tests als Sicherheitsnetz wandten sie systematisch „Methode extrahieren" an, um eine einzige gemeinsame Berechnungsfunktion zu erstellen, und ersetzten dann alle vierzig Aufrufstellen. Die Characterization Tests offenbarten drei Methoden mit echt unterschiedlichem Verhalten — beabsichtigte Spezialisierungen, die nie dokumentiert worden waren —, die das Team als explizit benannte Varianten bewahrte. Defekte bei der Währungsrundung fielen in den sechs Monaten nach dem Refactoring auf null.

Ein Telekommunikationsanbieter, der eine Legacy-Abrechnungsplattform betrieb, stellte fest, dass neue Ingenieure typischerweise vier bis sechs Monate brauchten, bevor sie die Rating-Engine sicher modifizieren konnten — ein dichtes Modul mit globalem Zustand, fest codierten Schwellenwerten und ohne Tests. Statt ein dediziertes Refactoring-Projekt zu planen — zwei vorherige Versuche waren abgebrochen worden, als sich Geschäftsprioritäten verschoben — bettete das Team Refactoring direkt in ihren Bugfix-Workflow ein. Jeder Defektfix in der Rating-Engine wurde von einem Refactoring-Commit vorangestellt, der die relevante Logik vom globalen Zustand isolierte. Über zwölf Monate transformierte die Praxis die Struktur der Rating-Engine ausreichend, dass die Einarbeitungszeit neuer Mitarbeiter für dieses Modul auf sechs Wochen sank, und das Team konnte ein lange verschobenes Mehrwährungs-Feature in drei Wochen einführen statt in den geschätzten drei Monaten.

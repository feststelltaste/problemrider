---
title: Große Pull Requests
description: Pull Requests sind zu groß, um wirksam reviewt zu werden, was zu oberflächlichen
  Reviews und übersehenen Problemen führt.
category:
- Code
- Communication
- Process
related_problems:
- slug: reduced-code-submission-frequency
  similarity: 0.75
- slug: rushed-approvals
  similarity: 0.75
- slug: extended-review-cycles
  similarity: 0.75
- slug: inadequate-code-reviews
  similarity: 0.7
- slug: code-review-inefficiency
  similarity: 0.7
- slug: review-process-breakdown
  similarity: 0.7
solutions:
- code-review-process-reform
- trunk-based-development
- small-change-batches
- code-review-guidelines
- feature-toggles
- incremental-refactoring
- continuous-integration
- lightweight-design-review
layout: problem
lang: de
en_slug: large-pull-requests
---

## Description

Große Pull Requests treten auf, wenn Entwickler Codeänderungen einreichen, die zu umfangreich oder komplex sind, damit Reviewer sie innerhalb angemessener Zeitbeschränkungen gründlich untersuchen können. Diese übergroßen Änderungen machen es praktisch unmöglich, aussagekräftige Code-Reviews durchzuführen, weil Reviewer das Review entweder ganz überspringen, nur oberflächliche Prüfungen durchführen oder Änderungen genehmigen, ohne ihre Implikationen vollständig zu verstehen. Große Pull Requests untergraben die primären Zwecke von Code-Review: Fehler erfassen, Wissen teilen und Codequalität wahren.

## Indicators ⟡
- Pull Requests enthalten regelmäßig Hunderte oder Tausende Zeilen an Änderungen
- Code-Reviews dauern ungewöhnlich lange oder werden sehr schnell ohne aussagekräftiges Feedback genehmigt
- Reviewer kommentieren häufig "LGTM" (Looks Good To Me) ohne substanzielle Review-Kommentare
- Entwickler vermeiden es, bestimmte Pull Requests aufgrund ihrer Größe und Komplexität zu reviewen
- Mehrere unzusammenhängende Features oder Fehlerbehebungen werden in einzelnen Pull Requests gebündelt

## Symptoms ▲

- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Reviewer, die mit übergroßen Pull Requests konfrontiert sind, greifen auf oberflächliche Prüfungen zurück und übersehen wichtige Design- und Logikprobleme.
- [Übereilte Genehmigungen](uebereilte-genehmigungen.md)
<br/>  Große PRs werden ohne gründliches Review genehmigt, weil Reviewern Zeit oder Energie fehlt, den vollständigen Änderungssatz zu untersuchen.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Große Pull Requests brauchen viel länger zum Reviewen, was Engpässe schafft, die die gesamte Entwicklungspipeline verzögern.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Fehler entkommen oberflächlichen Reviews großer PRs und erreichen die Produktion, was die Fehlerrate erhöht.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Wenn große Pull Requests wirksames Review umgehen, werden mehr Defekte unentdeckt in die Codebasis eingeführt.

## Causes ▼

- [Großer Feature-Umfang](grosser-feature-umfang.md)
<br/>  Features, die zu groß sind, um in inkrementelle Änderungen zerlegt zu werden, produzieren natürlich übergroße Pull Requests.
- [Langlebige Feature-Branches](langlebige-feature-branches.md)
<br/>  Branches, die über lange Zeiträume Änderungen anhäufen, resultieren in massiven Pull Requests, wenn sie schließlich zum Review eingereicht werden.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Wenn Entwickler Änderungen bündeln und selten einreichen, enthält jede Einreichung viel mehr Änderungen als nötig.

## Detection Methods ○
- **Pull-Request-Größenmetriken:** Nachverfolgung geänderter Codezeilen, modifizierter Dateien und Komplexitätsmetriken für Pull Requests
- **Review-Zeit-Analyse:** Überwachung, wie lange Reviews dauern, und Korrelation mit der Pull-Request-Größe
- **Bewertung der Review-Qualität:** Analyse der Tiefe und Qualität des Feedbacks zu unterschiedlich großen Pull Requests
- **Genehmigungsmuster:** Identifikation von Pull Requests, die im Verhältnis zu ihrer Größe schnell genehmigt werden
- **Entwicklerfeedback:** Befragung von Teammitgliedern zu ihrer Erfahrung beim Reviewen großer Pull Requests

## Examples

Ein Entwickler arbeitet drei Wochen isoliert an der Implementierung eines neuen Nutzerauthentifizierungssystems. Als er schließlich den Pull Request einreicht, enthält er 2.500 Zeilen neuen Codes über 45 Dateien hinweg, einschließlich Datenbankschema-Änderungen, neuer API-Endpunkte, Frontend-Komponenten, Konfigurationsaktualisierungen und Dokumentationsänderungen. Die zugewiesenen Reviewer betrachten den massiven Pull Request und geben entweder minimales Feedback ("insgesamt sieht gut aus") oder konzentrieren sich nur auf offensichtliche Probleme wie Code-Formatierung, wobei sie offensichtliche Fehler und architektonische Probleme übersehen. Aufgrund der Größe hat kein Reviewer die Zeit oder Energie, den kompletten Authentifizierungsfluss zu verstehen, zu verifizieren, dass Sicherheitsanforderungen erfüllt sind, oder sicherzustellen, dass die Implementierung etablierten Mustern folgt. Mehrere kritische Sicherheitslücken gelangen in die Produktion, weil sie im großen Änderungssatz begraben waren. Ein weiteres Beispiel betrifft einen Pull Request, der ein größeres Refactoring der Datenzugriffsschicht mit drei neuen Features und Fehlerbehebungen für zwei bestehende Features kombiniert. Der 1.800-Zeilen-Pull-Request umfasst mehrere Geschäftsdomänen und erfordert Expertise in unterschiedlichen Bereichen des Systems. Reviewer konzentrieren sich auf die Teile, die sie am besten verstehen, und überspringen Bereiche außerhalb ihrer Expertise, was zu Integrationsproblemen und inkonsistenter Codequalität über die unterschiedlichen Änderungen hinweg führt.

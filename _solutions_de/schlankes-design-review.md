---
title: Schlankes Design-Review
description: Besprechung des beabsichtigten Ansatzes für nicht-triviale
  Änderungen vor der Umsetzung, in einer kurzen Sitzung mit schriftlicher Skizze,
  sodass Designprobleme auftauchen, bevor Code existiert.
category:
- Architecture
- Code
- Process
problems:
- suboptimal-solutions
- complex-implementation-paths
- insufficient-design-skills
- second-system-effect
- rapid-prototyping-becoming-production
- quality-compromises
- large-pull-requests
- procedural-programming-in-oop-languages
- misunderstanding-of-oop
- process-design-flaws
- large-feature-scope
- convenience-driven-development
- god-object-anti-pattern
- accumulated-decision-debt
- analysis-paralysis
- communication-risk-within-project
- defensive-coding-practices
- delayed-decision-making
- inadequate-initial-reviews
- inexperienced-developers
- over-reliance-on-utility-classes
- poor-encapsulation
- tangled-cross-cutting-concerns
- unproductive-meetings
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: lightweight-design-review
related_solutions:
- slug: architecture-reviews
  similarity: 0.8
- slug: code-review-guidelines
  similarity: 0.75
- slug: code-review-process-reform
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.65
- slug: mikado-method
  similarity: 0.65
---

## Description

Ein schlankes Design-Review ist eine kurze Besprechung darüber, wie eine Änderung angegangen wird, abgehalten bevor die Änderung gebaut wird, basierend auf einer schriftlichen Skizze von einer Seite oder weniger. Es füllt eine Lücke, die die meisten Teams haben, ohne es zu bemerken: Code-Review untersucht eine bereits getroffene und teuer umzukehrende Entscheidung, während formales Architektur-Review zu schwerfällig ist, um für die gewöhnlichen Änderungen aufgerufen zu werden, bei denen die meisten Designentscheidungen tatsächlich getroffen werden. Die Konsequenz ist, dass sich der Großteil des Designs eines Legacy-Systems durch individuelle Entscheidungen ansammelt, die niemand besprochen hat. Der Mechanismus ist günstig, weil ein Design günstig zu ändern ist — eine Grenze in einer Skizze zu verschieben kostet Minuten, und sie in drei Wochen implementiertem Code zu verschieben kostet eine Verhandlung. Sein sekundärer Effekt zählt genauso: Es ist eine der wenigen Umgebungen, in denen Designbegründung explizit gemacht wird und daher gelernt werden kann.

## How to Apply ◆

> In Legacy-Arbeit ist die folgenreichste Designentscheidung meist, wo der neue Code relativ zum alten platziert wird — und diese Entscheidung wird in der ersten Stunde getroffen, allein, und nie besprochen.

- Definieren Sie einen **Auslöser**, damit klar ist, wann ein Review erwartet wird: eine Änderung über eine bestimmte Größe hinaus, eine, die eine neue Komponente oder Schnittstelle hinzufügt, eine, die mehr als ein Subsystem berührt, oder eine, die eine Abhängigkeit einführt. Ohne einen Auslöser geschieht es für die Änderungen, bei denen Menschen sich bereits unsicher fühlen, was nicht die riskanten sind.
- Verlangen Sie **eine schriftliche Skizze, auf eine Seite begrenzt**: was die Änderung tun muss, den vorgeschlagenen Ansatz, was in Betracht gezogen und abgelehnt wurde, und was betroffen sein wird. Der Akt, dies zu schreiben, fängt einen bedeutenden Anteil der Probleme ab, bevor irgendjemand sonst es liest.
- Halten Sie die Sitzung **kurz — dreißig Minuten — und klein**, zwei oder drei Personen einschließlich jemandem, der den betroffenen Bereich kennt. Ein großes Review wird zu einer Design-by-Committee-Sitzung und produziert Kompromissarchitekturen.
- Fokussieren Sie auf eine **kleine Menge an Fragen**: Passt das zu dem, wie das System bereits organisiert ist, was passiert, wenn die Teile, von denen es abhängt, ausfallen, wie wird es sein, es in zwei Jahren zu ändern, und gibt es einen einfacheren Ansatz. Alles Detailliertere gehört ins Code-Review.
- **Beziehen Sie die Option ein, es nicht so zu bauen.** Das wertvollste Ergebnis eines Design-Reviews ist oft die Entdeckung, dass ein bestehender Mechanismus bereits das meiste davon tut, und dieses Ergebnis ist nur vor der Umsetzung verfügbar.
- **Erfassen Sie die Entscheidung und die Begründung**, kurz, dort, wo sie später gefunden wird — idealerweise als Architecture Decision Record für alles Folgenreiche. Die Skizze plus das Ergebnis ist oft die einzige Design-Dokumentation, die die Änderung je haben wird.
- Nutzen Sie es bewusst als **Lehre**. Weniger erfahrene Entwickler, die ihren Ansatz präsentieren und die Fragen hören, die erfahrene Reviewer stellen, ist es, wie sich Designurteilsvermögen überträgt; es wird sonst im Wesentlichen nie explizit gelehrt.
- **Lassen Sie es nicht zu einem Gate werden.** Ein Review, das Arbeit blockiert, während auf den Kalender eines leitenden Reviewers gewartet wird, wird umgangen, und zwar zu Recht. Reaktion am selben oder nächsten Tag ist die Anforderung, mit einem erklärten Fallback für den Fall, dass es nicht geschehen kann.
- **Halten Sie es aus trivialen Fällen heraus.** Auf jede Änderung angewendet, wird der Overhead übel genommen, und die Praxis wird zusammen mit den Fällen aufgegeben, in denen sie wertvoll war.

## Tradeoffs ⇄

> Das Design vor der Umsetzung zu überprüfen fängt Probleme ab, während sie günstig sind, zum Preis einer Verzögerung, bevor das Codieren beginnt, und einer Praxis, die leicht bürokratisch wird.

**Vorteile:**

- Designprobleme werden abgefangen, wenn ihre Änderung Minuten statt Wochen kostet, was das gesamte ökonomische Argument ist.
- Duplizierte Mechanismen werden vermieden, weil jemand im Raum meist weiß, dass das System dies bereits irgendwo tut.
- Pull Requests werden kleiner und besser überprüfbar, da der Ansatz feststeht und das Code-Review sich auf Korrektheit fokussieren kann.
- Designbegründung wird explizit und beobachtbar, was der Weg ist, wie sie sich zu Entwicklern verbreitet, denen sie zuvor nicht beigebracht wurde.
- Die Konsistenz über das System hinweg verbessert sich, weil unabhängige Änderungen gegen die bestehende Organisation geprüft werden, statt jede ihre eigene zu erfinden.

**Kosten und Risiken:**

- Die Umsetzung beginnt später, und für Änderungen, die in Ordnung gewesen wären, ist die Verzögerung reine Kosten.
- Die Praxis driftet in Richtung eines formalen Genehmigungs-Gates, an welchem Punkt sie zu einem Hindernis wird und von wem auch immer es eilig hat, umgangen wird.
- Designdiskussion kann zu Überkonstruktion führen, besonders wenn die Reviewer erfahrener sind, als das Problem erfordert, und die Diskussion genießen.
- Reviews mit zu vielen Teilnehmern konvergieren auf das Design, das niemanden beleidigt, was häufig schlimmer ist als beide Alternativen.
- Die Skizze kann zu einem schwergewichtigen Dokument werden, wenn Standards schleichend wachsen, und eine Ein-Seiten-Anforderung braucht aktive Verteidigung.

## How It Could Be

Ein Team, das ein Dokumentenmanagementsystem pflegte, produzierte konsistent Pull Requests von 800 bis 2.000 Zeilen, die Reviewer nur abnicken statt bewerten konnten, und etwa ein Viertel davon löste nach dem Review erhebliche Nacharbeit aus. Sie führten einen Auslöser ein — alles, was voraussichtlich mehr als drei Tage dauern würde, oder das mehr als ein Subsystem berührt —, der eine Ein-Seiten-Skizze und eine dreißigminütige Diskussion erforderte. Die erste überprüfte Skizze schlug einen neuen Hintergrund-Worker für Thumbnail-Erzeugung vor. Innerhalb von zehn Minuten wies ein Kollege darauf hin, dass das bestehende Batch-Framework bereits Terminierung, Retries und Fehlerwarnung handhabte, wovon nichts der Vorschlag berücksichtigt hatte. Die Änderung ging von geschätzten zwei Wochen auf vier Tage. Über die folgenden sechs Monate sank der Anteil der Pull Requests, die nach dem Review erhebliche Nacharbeit erforderten, von etwa einem Viertel auf unter fünf Prozent.

Der Lehreffekt erwies sich als der dauerhaftere. Zwei Entwickler mit Hintergrund in prozeduralen Codebasen hatten konsistent Designs produziert, die Logik in statischen Helper-Klassen platzierten, wogegen Reviewer dann beim Code-Review Einwände erhoben — nachdem der Code geschrieben war, was das Gespräch gegnerisch machte. Im Design-Review kam derselbe Einwand als Frage danach an, wohin das Verhalten gehörte, bevor irgendein Code existierte, und konnte als echte Frage diskutiert werden. Innerhalb weniger Monate stellten beide Entwickler diese Frage selbst in ihren Skizzen. Die Reviewer des Teams bemerkten, dass sie nun Begründungen erklärten statt Änderungen zu verlangen, was ein materiell anderes Gespräch war.

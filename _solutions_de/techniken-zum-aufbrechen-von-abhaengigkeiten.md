---
title: Techniken zum Aufbrechen von Abhängigkeiten
description: Schaffung von Nahtstellen in nicht testbarem Code — extrahieren, umhüllen,
  parametrisieren, subklassieren — sodass ein Fragment isoliert und ohne den Rest
  des Systems ausgeführt werden kann.
category:
- Code
- Testing
- Architecture
problems:
- difficult-to-test-code
- monolithic-functions-and-classes
- global-state-and-side-effects
- testing-complexity
- poor-encapsulation
- excessive-class-size
- bloated-class
- over-reliance-on-utility-classes
- tight-coupling-issues
- high-coupling-low-cohesion
- legacy-code-without-tests
- hidden-dependencies
- flaky-tests
- circular-references
- god-object-anti-pattern
- refactoring-avoidance
- test-debt
- complex-implementation-paths
- hidden-side-effects
- maintenance-paralysis
- brittle-codebase
layout: solution
lang: de
en_slug: dependency-breaking-techniques
related_solutions:
- slug: mikado-method
  similarity: 0.7
- slug: test-coverage-strategy
  similarity: 0.7
- slug: characterization-tests
  similarity: 0.7
- slug: modularization-and-bounded-contexts
  similarity: 0.7
- slug: regression-testing
  similarity: 0.65
- slug: change-impact-analysis
  similarity: 0.65
---

## Description

Techniken zum Aufbrechen von Abhängigkeiten sind eine Reihe kleiner, mechanischer Code-Transformationen, deren Zweck es ist, eine Nahtstelle zu schaffen — eine Stelle, an der Verhalten ersetzt werden kann, ohne den Code zu bearbeiten, der es nutzt. Sie existieren, um die Zirkularität aufzulösen, die Legacy-Arbeit definiert: Der Code kann wegen seiner Abhängigkeiten nicht getestet werden, und die Abhängigkeiten können ohne Tests nicht sicher entfernt werden. Jede Technik ist bewusst minimal und risikoarm, gestaltet, um von Hand ohne Test-Sicherheitsnetz angewandt zu werden, weil in dem Moment, in dem Sie sie brauchen, kein Test-Sicherheitsnetz existiert. Die auszeichnende Eigenschaft ist Konservatismus: Eine Methode extrahieren, einen Parameter hinzufügen oder subklassieren, um einen einzelnen Aufruf zu überschreiben, sind Änderungen, die ein sorgfältiger Entwickler durch Lesen verifizieren kann. Sobald eine Nahtstelle existiert, kann ein Test durch sie hindurch geschrieben werden, und ab diesem Punkt wird gewöhnliches Refactoring verfügbar.

## How to Apply ◆

> Dies sind die Techniken für die spezifische Situation, in der die Instanziierung einer Klasse eine Datenbankverbindung, einen Message Broker, einen Lizenzserver und die Systemuhr mitzieht.

- **Extrahieren und überschreiben**: Ziehen Sie den problematischen Aufruf in eine eigene protected Methode, erstellen Sie dann eine testexklusive Unterklasse, die sie überschreibt. Dies ist die am breitesten anwendbare Technik und erfordert fast keine Änderung am ursprünglichen Code — die Extraktion ist mechanisch, und die Überschreibung lebt vollständig im Test.
- **Konstruktor oder Methode parametrisieren**: Wo eine Klasse ihren eigenen Mitarbeiter intern konstruiert, fügen Sie einen Parameter hinzu, der einen akzeptiert, standardmäßig auf die ursprüngliche Konstruktion zurückfallend. Bestehende Aufrufer sind unbeeinflusst, und Tests können jetzt einen Ersatz liefern. Den Standard beizubehalten bewahrt Verhalten für alle bestehenden Aufrufstellen, was dies sicher macht, ohne Tests zu tun.
- **Führen Sie eine Schnittstelle an der Grenze ein** und lassen Sie die Legacy-Klasse sie implementieren. Aufrufer hängen von der Schnittstelle ab; Tests liefern eine andere Implementierung. Dies ist der Standardzug zur Isolierung von Datenbanken, Dateisystemen, externen Services und der Uhr.
- **Sprout-Methode oder -Klasse**: Wenn Sie neues Verhalten zu einer verworrenen Methode hinzufügen, schreiben Sie es in eine neue, vollständig testbare Methode oder Klasse, und lassen Sie den Legacy-Code sie aufrufen. Der Legacy-Code wächst um eine Zeile; die neue Logik ist von Anfang an getestet. Dies verbessert den alten Code nicht, verhindert aber, dass der neue Code Teil des Problems wird.
- **Methode oder Klasse umhüllen**: Um Verhalten um bestehenden Code herum hinzuzufügen — Logging, Validierung, eine Metrik —, benennen Sie das Original um und erstellen Sie einen Wrapper mit dem alten Namen, der es aufruft. Aufrufer bleiben unverändert, und der Wrapper ist unabhängig testbar.
- **Brechen Sie eine statische oder globale Abhängigkeit auf**, indem Sie eine instanzebenenbasierte Indirektion einführen: Ersetzen Sie direkte Aufrufe eines statischen Halters durch Aufrufe eines Felds, das standardmäßig auf den statischen Halter zurückfällt. Globaler Zustand ist meist das einzige größte Hindernis beim Testen von Legacy-Code, und dies verwandelt es von einem Hindernis in ein ersetzbares.
- **Kapseln Sie die Uhr, Zufälligkeit und die Umgebung** früh hinter Schnittstellen und behandeln Sie jeden direkten Aufruf von `now()` oder einem Zufallsgenerator als Defekt. Diese drei Abhängigkeiten machen einen unverhältnismäßigen Anteil nicht testbaren und zeitweise fehlschlagenden Verhaltens aus.
- Wenden Sie die Techniken **einzeln an und verifizieren Sie durch Lesen**. Jede Transformation sollte klein genug sein, dass ihre Verhaltensbewahrung selbstverständlich ist. Wenn Sie sich durch Lesen nicht überzeugen können, dass eine Änderung verhaltensneutral ist, ist es ein zu großer Schritt.
- Sobald eine Nahtstelle existiert, **schreiben Sie sofort einen Charakterisierungstest durch sie hindurch**, bevor Sie weitere Änderungen vornehmen. Die Nahtstelle hat keinen Wert, bis etwas sie ausübt, und das Fenster, in dem der Code verstanden ist, ist kurz.

## Tradeoffs ⇄

> Diese Techniken kaufen Testbarkeit auf Kosten etwas zusätzlicher Indirektion, und ohne Richtung angewandt erzeugen sie eine Codebasis voller Nahtstellen, die nichts nutzt.

**Vorteile:**

- Code, der nicht testbar war, wird testbar, was jede nachfolgende Verbesserung freischaltet — Refactoring, sichere Fehlerbehebung und eventuelle Extraktion oder Ersatz.
- Die Transformationen sind einzeln risikoarm und durch Inspektion überprüfbar, sodass sie auf Code ohne Testabdeckung angewandt werden können, was die Situation ist, in der sie gebraucht werden.
- Über Sprout-Techniken hinzugefügte neue Funktionalität ist von Anfang an getestet, sodass der Anteil getesteten Codes steigt, selbst wenn der Legacy-Teil nie adressiert wird.
- Das Isolieren von Zeit, Zufälligkeit und externen Services entfernt typischerweise einen erheblichen Anteil intermittierender Testfehler und nicht reproduzierbarer Defekte.
- Zum Testen eingeführte Nahtstellen werden häufig zu den natürlichen Grenzen für spätere Extraktion, sodass die Arbeit nicht verschwendet ist, wenn eine Strangler-Fig-Migration folgt.

**Kosten und Risiken:**

- Jede Technik fügt eine Schicht Indirektion hinzu. Großzügig ohne Ziel angewandt, ist das Ergebnis eine Codebasis, die schwerer zu lesen ist als das Original, während sie nur marginal besser getestet ist.
- Extrahieren-und-Überschreiben insbesondere erzeugt testexklusive Unterklassen und protected Methoden, die allein zum Testen existieren, was manche Teams unattraktiv finden und das Testbelange in Produktionscode durchsickern lässt.
- Ohne Tests trägt jede Transformation ein kleines Restrisiko einer Verhaltensänderung, und das Risiko häuft sich über viele Transformationen an. Disziplin bei der Schrittgröße ist essenziell.
- Die Techniken adressieren Struktur, nicht Design. Eine testbar gemachte Klasse ist dadurch nicht wohl gestaltet, und Teams hören manchmal bei Testbarkeit auf und betrachten das Modul als adressiert.
- Legacy-Sprachen und -Frameworks variieren darin, wie gut sie diese Züge unterstützen; manche machen Ersatz echt schwierig, und der Aufwand kann den Wert für ein selten geändertes Modul übersteigen.

## How It Could Be

Ein Entwickler musste einen Rundungsdefekt in einer Bestellsummenberechnung beheben. Die berechnende Klasse konstruierte eine Datenbankverbindung, las eine statische Währungskonfiguration und rief die Systemuhr für das Wechselkursdatum auf — sie in einem Test zu instanziieren war unmöglich. Statt zu versuchen, die Klasse umzustrukturieren, wandte sie in einem Nachmittag drei Transformationen an: Die Wechselkursabfrage wurde in eine protected Methode extrahiert und in einer Test-Unterklasse überschrieben, die Währungskonfiguration wurde durch ein Feld ersetzt, das standardmäßig auf den statischen Halter zurückfiel, und der Uhr-Aufruf wurde zu einem Konstruktorparameter mit Standardwert. Keine der Änderungen veränderte das Produktionsverhalten, und alle drei waren durch Lesen verifizierbar. Sie schrieb dann elf Charakterisierungstests, fand heraus, dass der Rundungsdefekt einer von dreien war, und behob alle drei mit Vertrauen.

Ein Team, das eine neue Betrugsprüfung zu einem Zahlungsablauf hinzufügte, entschied sich für Sprouting statt Modifikation. Die bestehende Zahlungsmethode hatte 700 Zeilen ohne Tests; statt sie zu erweitern, schrieben sie eine `FraudCheck`-Klasse mit vollständiger Testabdeckung und fügten einen einzigen Aufruf in die Legacy-Methode ein. Die Legacy-Methode wuchs um eine Zeile und blieb so ungetestet wie zuvor, aber die neue Logik — der Teil, der sich am wahrscheinlichsten ändern musste, während sich Betrugsmuster weiterentwickelten — war von Tag eins ordentlich getestet. Über die folgenden zwei Jahre wurde die Betrugsprüfung vierzehnmal modifiziert, immer sicher, während die umgebende Legacy-Methode nie angefasst wurde.
